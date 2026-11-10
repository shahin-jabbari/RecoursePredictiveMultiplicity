from experiment import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn

def fit_candidate(root,ds,seed,stream,index,a,folder):
 p=read(root/'protocol.json');s=seed_for(p['arm'],ds,seed,stream,index)
 X=a['X_train'];y=a['y_train'];outer=None
 if stream==2:
  outer=np.random.default_rng(s+500_000_000).integers(0,len(y),len(y));X=X[outer];y=y[outer]
 kwargs=dict(max_depth=4,min_samples_leaf=5,max_features='sqrt',random_state=s)
 clf=(RandomForestClassifier(n_estimators=10,bootstrap=True,n_jobs=1,**kwargs) if p['arm']=='random_forest_pilot' else DecisionTreeClassifier(**kwargs)).fit(X,y)
 native=clf.predict_proba(a['X_validation'])[:,list(clf.classes_).index(1)]
 model=rt.TreeModel.from_estimator(clf,metadata=dict(fit_seed=s,architecture=kwargs,forest_trees=10 if p['arm']=='random_forest_pilot' else 1,outer_bootstrap_seed=s+500_000_000 if stream==2 else None))
 assert np.allclose(model.predict_probability(a['X_validation']),native,rtol=0,atol=1e-12)
 model.save(folder)
 if outer is not None:np.save(folder/'bootstrap_indices.npy',outer,allow_pickle=False)
 return model,dict(index=index,fit_seed=s,validation_accuracy=float(((native>=.5)==a['y_validation']).mean()),fingerprint=fingerprint(model),tree_count=len(model.trees),nodes=sum(len(t.left) for t in model.trees))

def prepare(root):
 root=Path(root);p=read(root/'protocol.json');begin=time.perf_counter()
 assert not (root/'freeze.json').exists(),'No automatic preparation resume'
 source=Path(p['source_root']);assert read(source/'verification.json')['passed'] and read(source/'worker_exit.json')['completed']
 preservation={};inventory=[];cohorts=read(source/'cohorts.json');write(root/'cohorts.json',cohorts)
 for n in ['records.jsonl','cohorts.json','protocol.json','verification.json','worker_exit.json']:
  preservation[str(source/n)]=sha(source/n)
 if p['arm']=='decision_tree':
  shutil.copy2(source/'records.jsonl',root/'records.jsonl')
  assert sha(root/'records.jsonl')==preservation[str(source/'records.jsonl')]
  source_rows=rows(root/'records.jsonl');assert len(source_rows)==25200 and len({r['record_id'] for r in source_rows})==25200
  plan=dict(expected_method_rows=25200,expected_evaluation_rows=50400,expected_banks=60,expected_people=300,expected_unseen_models=6000,new_recourse_calls=0)
 else:
  cells=make_cells(p);assert len(cells)==72 and sum(c['budget']>0 for c in cells)==48
  plan=dict(cells=cells,expected_method_rows=288,expected_evaluation_rows=576,expected_banks=12,expected_people=12,expected_unseen_models=1200,positive_cells=48,zero_cells=24,expected_rounds=4800,expected_shared_reference_tasks=24,full_launch_authorized=False)
 write(root/'task_plan.json',plan)
 for ds in p['datasets']:
  for seed in p['seeds']:
   start=time.perf_counter();stage(root,'prepare_bank',dataset=ds,seed=seed,banks_completed=len(inventory));progress(root,status='preparing',arm=p['arm'],banks_completed=len(inventory),expected_banks=plan['expected_banks'],completed_method_rows=0)
   original=source/'banks'/ds/f'seed_{seed}';b=root/'banks'/ds/f'seed_{seed}';b.mkdir(parents=True)
   for n in ['split.npz','bank.json','admission.json']:
    preservation[str(original/n)]=sha(original/n);shutil.copy2(original/n,b/('source_'+n if n!='split.npz' else n))
   a=np.load(b/'split.npz',allow_pickle=False);meta=read(original/'bank.json');seen=[]
   if p['arm']=='decision_tree':
    shutil.copy2(original/'bank.json',b/'bank.json');shutil.copy2(original/'admission.json',b/'admission.json')
    for i in meta['selected_candidate_indices']:
     src=original/'candidates'/f'{i:04d}';dst=b/'candidates'/f'{i:04d}';shutil.copytree(src,dst)
     for file in src.iterdir():
      if file.is_file():preservation[str(file)]=sha(file)
     seen.append(rt.TreeModel.load(dst))
    cutoff=read(b/'admission.json')['minimum_admissible_accuracy']
   else:
    entries=[];fps=set();selected=[];reference=None
    for i in range(p['competing_candidate_limit']):
     model,e=fit_candidate(root,ds,seed,0,i,a,b/'candidates'/f'{i:04d}');e['first_unique']=e['fingerprint'] not in fps;fps.add(e['fingerprint']);entries.append(e)
     if i>=95:
      if reference is None:reference=max(z['validation_accuracy'] for z in entries[:96])
      cutoff=reference-.05;selected=[z['index'] for z in entries if z['first_unique'] and z['validation_accuracy']>=cutoff-1e-12][:24]
      if len(selected)==24:break
     if (i+1)%32==0:stage(root,'train_forest_bank',dataset=ds,seed=seed,candidates=i+1)
    write(b/'admission.json',dict(reference_candidate_count=96,best_validation_accuracy=reference,minimum_admissible_accuracy=cutoff,candidates=entries,selected=selected,reserve_limit=p['competing_candidate_limit']))
    if len(selected)!=24:raise RuntimeError(f'Fixed forest candidate reserve exhausted: {ds}/{seed}: {len(selected)}/24; no cutoff relaxation')
    meta=dict(meta,selected_candidate_indices=selected,best_validation_accuracy=reference,family='random_forest',trees_per_model=10)
    write(b/'bank.json',meta);seen=[rt.TreeModel.load(b/'candidates'/f'{i:04d}') for i in selected]
   # Factual cohort is exactly the old five-person cohort; pilot uses its first person.
   positions=cohorts[ds][str(seed)]['positions'];assert len(positions)==5
   assert cohorts[ds][str(seed)]['source_ids']==[int(a['test_ids'][i]) for i in positions]
   factual=a['X_test'][positions];needed=common_eligibility(seen,factual)
   write(b/'cohort_audit.json',dict(positions=positions,source_ids=cohorts[ds][str(seed)]['source_ids'],common_recourse_needed=needed,seen_factual_probabilities=independent_prob(seen,factual)))
   for model in seen:
    assert np.allclose(model.predict_probability(a['X_test']),native_saved_prediction(model,a['X_test']),rtol=0,atol=1e-12)
   exclude={fingerprint(m) for m in seen};panel_info={}
   for stream,panel in [(1,'refit'),(2,'bootstrap')]:
    panel_start=time.perf_counter();folder=b/'panels'/panel;entries=[];selected=[];fps=set(exclude)
    for i in range(p['unseen_candidate_limit']):
     model,e=fit_candidate(root,ds,seed,stream,i,a,folder/'candidates'/f'{i:04d}')
     e['distinct_from_seen_and_prior_candidates']=e['fingerprint'] not in fps;fps.add(e['fingerprint'])
     e['accepted']=e['distinct_from_seen_and_prior_candidates'] and e['validation_accuracy']>=cutoff-1e-12
     entries.append(e)
     if e['accepted']:selected.append(i)
     if len(selected)==50:break
     if (i+1)%100==0:stage(root,'train_unseen_panel',dataset=ds,seed=seed,panel=panel,accepted=len(selected),candidates=i+1)
    info=dict(panel=panel,selected=selected,candidates=entries,minimum_admissible_accuracy=cutoff,seen_fingerprints=sorted(exclude),selection='first 50 distinct eligible new-seed models, no action/prediction filtering',candidate_limit=p['unseen_candidate_limit'])
    write(folder/'admission.json',info)
    if len(selected)!=50:raise RuntimeError(f'Fixed unseen candidate reserve exhausted: {ds}/{seed}/{panel}: {len(selected)}/50; no cutoff relaxation')
    panel_models=[rt.TreeModel.load(folder/'candidates'/f'{i:04d}') for i in selected]
    native_test=np.stack([native_saved_prediction(m,a['X_test']) for m in panel_models],axis=1)
    native_val=np.stack([native_saved_prediction(m,a['X_validation']) for m in panel_models],axis=1)
    assert np.allclose(independent_prob(panel_models,a['X_test']),native_test,rtol=0,atol=1e-12)
    np.savez_compressed(folder/'native_prediction_audit.npz',test=native_test,validation=native_val)
    panel_info[panel]=dict(candidates=len(entries),selected=50,validation_mean=float(np.mean([entries[i]['validation_accuracy'] for i in selected])))
    append(root/'timing.jsonl',dict(kind='unseen_panel',dataset=ds,seed=seed,panel=panel,candidates=len(entries),seconds=time.perf_counter()-panel_start,utc=now()))
   entry=dict(dataset=ds,seed=seed,models=24,panels=panel_info,selected_people=5 if p['arm']=='decision_tree' else 1,source_positions=positions,common_eligible_count=int(needed.sum()))
   inventory.append(entry);append(root/'timing.jsonl',dict(kind='preparation_bank',dataset=ds,seed=seed,seconds=time.perf_counter()-start,includes_unseen_panels=True,utc=now()))
   progress(root,banks_completed=len(inventory),unseen_models_completed=len(inventory)*100);write(root/'model_inventory.json',inventory)
 write(root/'source_preservation.json',dict(sha256=preservation,original_method_rows_preserved=p['arm']=='decision_tree'))
 frozen=list((root/'source').rglob('*.py'))+list((root/'banks').rglob('*'))+[root/n for n in ['protocol.json','cohorts.json','task_plan.json','model_inventory.json','source_preservation.json']]
 if p['arm']=='decision_tree':frozen.append(root/'records.jsonl')
 write(root/'freeze.json',dict(utc=now(),before_evaluation=True,before_new_recourse=True,sha256=manifest(root,frozen)))
 # Independent admission and native prediction checks run before any action scoring/new recourse.
 from verify_unseen import verify_banks
 verify_banks(root)
 write(root/'preparation_verification.json',dict(passed=True,utc=now(),banks=len(inventory),panels_frozen=True,cohorts_preserved=True,seconds=time.perf_counter()-begin))
 stage(root,'ALL_INPUTS_FROZEN',banks=len(inventory),unseen_models=len(inventory)*100)
 return plan
