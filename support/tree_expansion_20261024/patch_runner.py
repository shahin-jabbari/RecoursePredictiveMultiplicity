from pathlib import Path
W=Path(__file__).resolve().parent;f=W/'source/runner.py';s=f.read_text()
s=s.replace('from prepare import prepare','from prepare import prepare,link\nfrom reuse import restore_exclusions,import_reference,verify_preservation')
s=s.replace("out.mkdir(parents=True);historical_seconds=0.;oracle_subdir='orpm_solver';diagnostic_subdir='diagnostic_solver'", """out.mkdir(parents=True);historical_seconds=0.;oracle_subdir='orpm_solver';diagnostic_subdir='diagnostic_solver'
 reuse=read(root/'reuse_manifest.json');prefix=reuse['prefixes'].get(key);saved=[]
 if prefix:
  src=Path(prefix['directory']);saved=rows(src/'rounds.jsonl');assert len(saved)==100 and sha(src/'rounds.jsonl')==prefix['sha256']
  link(src/'rounds.jsonl',out/'source_prefix.jsonl');link(src/'orpm_solver',out/'saved_orpm_solver')
  write(out/'reuse.json',dict(prefix=prefix,source_bytes_retained=True,native_optimizer_rebuilt=True))""")
s=s.replace("buildtime=time.perf_counter()-begin", """if saved:
  restore_exclusions(oracle,saved);oracle.calls=len(saved)
 buildtime=time.perf_counter()-begin""")
s=s.replace("w=np.exp(logw);x,info=checked_solve(oracle,w,known_points=(best,))\n   info['artifact_subdir']=oracle_subdir;l=loss(models,x);solve_seconds+=info['solver_seconds']", """w=np.exp(logw)
   if t<=len(saved):
    prior=saved[t-1];assert np.allclose(w,prior['weights'],atol=1e-12,rtol=0)
    x=torch.as_tensor(prior['point'],dtype=torch.float32);info=dict(prior['oracle']);info['artifact_subdir']='saved_orpm_solver';l=loss(models,x)
    assert np.allclose(l,prior['losses'],atol=1e-12,rtol=0)
   else:
    x,info=checked_solve(oracle,w,known_points=(best,));info['artifact_subdir']=oracle_subdir;l=loss(models,x);solve_seconds+=info['solver_seconds']""")
s=s.replace("if info['lower_bound']>float(cached[k])+1e-7:\n     au=", "if info['lower_bound']>float(cached[k])+1e-7:\n     if t<=len(saved):raise RuntimeError('Preserved prefix lower bound is inconsistent; no silent mutation')\n     au=")
s=s.replace("if t%10==0 or t==1 or t==148:","""if t<=len(saved):
    for field in ['round','weights','point','losses','best_point','best_losses','best_cost','joint_valid']:
     assert clean(row[field])==prior[field],('prefix_field_changed',field,t)
   if t%10==0 or t==1 or t==148:""")
s=s.replace("completed_rounds=(ordinal-1)*200+t,expected_rounds=expected*200,current=stage_data", "completed_rounds=(ordinal-1)*200+t,preserved_complete_trajectories=1620,preserved_prefix_rounds=162900,new_orpm_calls=new_calls_before(root,ordinal)+max(0,t-len(saved)),expected_new_orpm_calls=1673100,expected_rounds=expected*200,current=stage_data")
a=s.index(" refstart=time.perf_counter();reference=");z=s.index(" # Portable safe export",a)
s=s[:a]+""" refstart=time.perf_counter()
 if key in reuse['references']:
  ref=import_reference(root,out,models,x0,c,b,best,reuse['references'][key]);rx=torch.as_tensor(ref['point'],dtype=torch.float32)
 else:
  reference=rt.TreeOracle(models,x0,b,c,time_limit=30,abs_gap=1e-5,seed=seed,artifact_dir=out/'reference_solver')
  try:
   rx,ri=checked_solve(reference,objective='minimax',known_points=(best,))
   ref=dict(point=rx.numpy(),losses=loss(models,rx),oracle=ri,seconds=time.perf_counter()-refstart,orpm_warm_start=False)
  finally:
   if getattr(reference,'excluded_budget_leaf_combinations',None):reference.problem.write(str(out/'reference_solver/final_formulation.mps'))
   reference.close()
 write(out/'reference.json',ref)
"""+s[z:]
s=s.replace("for p in out.glob('*solver*/*.mps'):","for p in out.glob('*solver*/*.mps'):\n  if p.parent.is_symlink():continue")
s=s.replace("total_seconds=time.perf_counter()-begin+historical_seconds)", "total_seconds=time.perf_counter()-begin+historical_seconds,new_orpm_calls=200-len(saved),preserved_orpm_calls=len(saved),reused_reference=key in reuse['references'],new_reference_calls=int(key not in reuse['references']))")
s=s.replace("completed_rounds=ordinal*200,expected_rounds=expected*200,utc=now()", "completed_rounds=ordinal*200,preserved_complete_trajectories=1620,preserved_prefix_rounds=162900,new_orpm_calls=new_calls_before(root,ordinal+1),expected_new_orpm_calls=1673100,expected_rounds=expected*200,utc=now()")
s=s.replace("def run(root):", """def new_calls_before(root,ordinal):
 reuse=read(root/'reuse_manifest.json');plan=read(root/'task_plan.json')
 return sum(0 if z['key'] in reuse['complete'] else 100 if z['key'] in reuse['prefixes'] else 200 for z in plan['cells'][:ordinal-1])

def run(root):""")
s=s.replace("for ordinal,cell in enumerate(plan['cells'],1):\n  trajectory", "for ordinal,cell in enumerate(plan['cells'],1):\n  if cell['key'] in read(root/'reuse_manifest.json')['complete']:continue\n  trajectory")
s=s.replace(" verify(root)\n stage(root,'analysis_and_reporting')", " verify(root)\n verify_preservation(root)\n stage(root,'analysis_and_reporting')")
s=s.replace("expected_rounds=plan['expected_rounds'],utc=now()", "expected_rounds=plan['expected_rounds'],new_orpm_calls=plan['new_orpm_calls'],preserved_rounds=plan['preserved_rounds'],utc=now()")
s=s.replace('TREE_CONVERGENCE_MAIN_COMPLETED','TREE_CONVERGENCE_EXPANSION_COMPLETED')
f.write_text(s)
