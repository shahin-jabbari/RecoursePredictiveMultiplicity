import csv, html, shutil
from collections import Counter
from pathlib import Path
import numpy as np
from common import *

def prepare(root,droot):
 root=Path(root);droot=Path(droot);root.mkdir(parents=True,exist_ok=False);droot.mkdir(parents=True,exist_ok=False)
 source_config=read(TRADE/'config.json');assert read(TRADE/'verification.json')['status']=='passed'
 cfg=dict(version='1.0',study='neural_cost_and_disparity_reuse_timed_pilot',source_run=str(TRADE),disparity_result=str(droot),datasets=DATASETS,
  source_seeds=list(range(401,411)),pilot_seeds=[401,402,403],pilot_people_per_seed=2,model_counts=[6,12,24],budgets=source_config['budgets'],groups=GROUPS,
  maximum_optimization_seconds_per_solve=10.,threads=1,seed=0,expected_cases=36,expected_cost_comparisons=108,expected_tasks=972,
  source_cases=1200,source_orpm_settings=26400,full_cost_tasks=32400,training=False,new_unseen_models=False,
  success_logit_threshold=0.,budget_tolerance=1e-8,cost_certificate_tolerance='1e-4*(1+maximum_budget)',
  inference='pointwise descriptive Student t intervals across seed means; pilot n=3 df=2; reused cohort n=10 df=9; numerical bounds separate',
  scope='saved neural-only heterogeneous architecture sets; twelve 20-ReLU and twelve 50/100-ReLU networks, balanced at each nested size',
  cohort_rule='For each of seeds 401-403: first saved individual from each group if both represented; otherwise first two saved individuals; Synthetic/Polish first two. No new eligibility or cohort expansion.',
  structural_estimands='capped singleton, hardest singleton and common-action minimum standardized L2 at dataset maximum budget; cost and success partial-identification bounds',
  limits='10 seconds per actual optimization, plus encoding/export/I/O. No automatic full expansion. All model fits and ORPM actions reused unchanged.',
  expected_maximum_optimization_hours=972*10/3600)
 save(root/'protocol.json',cfg)
 shutil.copytree(Path(__file__).parent,root/'source',ignore=shutil.ignore_patterns('__pycache__'))
 hashes={str(TRADE/n):sha(TRADE/n) for n in ['protocol.json','config.json','verification.json','model_inventory.json','results.csv']}
 cases=[];full_cases=[];group_counts=[];inventory={};source_demand=[]
 for ds in DATASETS:
  for seed in range(401,411):
   old=TRADE/'blocks'/ds/f'seed_{seed}';selection=read(old/'selection.json');bank=Path(selection['source_bank'])
   pre=read(bank/'preprocessing.json');meta=read(bank/'models.json')
   for p in [old/'selection.json',old/'cohort.npz',bank/'models.json',bank/'preprocessing.json',bank/'split_and_predictions.npz']:
    hashes[str(p)]=sha(p)
   with np.load(bank/'split_and_predictions.npz',allow_pickle=False) as z:X=z['X_test'][selection['test_positions']].astype(np.float32).astype(float)
   labs=labels(X,pre,ds);counts=Counter(labs.tolist())
   for g in sorted(counts):group_counts.append(dict(dataset=ds,seed=seed,group=g,saved_people=counts[g]))
   for i,iid in enumerate(selection['instance_ids']):full_cases.append(dict(dataset=ds,seed=seed,instance_id=int(iid),group=str(labs[i]),group_weight=counts[str(labs[i])]/len(X)))
   nets={}
   for j in range(24,48):
    p=bank/f'model_{j:04d}.npz';hashes[str(p)]=sha(p);nets[j]=load_net(p)
    assert meta['models'][j]['training_settings']['passed']
    assert [list(w.shape)[::-1] for w,b in nets[j]]==meta['models'][j]['linear_dimensions']
   zz=logits([nets[j] for j in range(24,48)],X)
   for architecture,slots in [('shallow',range(12)),('deep',range(12,24))]:
    source_demand.append(dict(dataset=ds,seed=seed,kind='singleton_'+architecture,tasks=240,nontrivial=int(np.sum(zz[:,list(slots)]<0))))
   for n in [6,12,24]:source_demand.append(dict(dataset=ds,seed=seed,kind='joint_'+str(n),tasks=20,nontrivial=int(np.sum(np.min(zz[:,np.array(indices(n))-24],axis=1)<0))))
   inventory[f'{ds}/{seed}']=dict(source_bank=str(bank),models=24,source_indices=list(range(24,48)),architecture_counts={'20_relu':12,'50_100_relu':12},retrained=False)
   if seed not in cfg['pilot_seeds']:continue
   if ds in GROUPS and len(counts)==2:chosen=[int(np.flatnonzero(labs==GROUPS[ds][k])[0]) for k in ['first','second']]
   else:chosen=[0,1]
   bdest=root/'blocks'/ds/f'seed_{seed}'/'bank';bdest.mkdir(parents=True)
   for name in ['models.json','preprocessing.json']+[f'model_{j:04d}.npz' for j in range(24,48)]:shutil.copy2(bank/name,bdest/name)
   for position in chosen:
    iid=int(selection['instance_ids'][position]);key=f'{ds}_{seed}_{iid}';person=old/f'person_{iid}';dest=root/'cases'/key;dest.mkdir(parents=True)
    f=person/'factual.npz';hashes[str(f)]=sha(f);shutil.copy2(f,dest/'factual.npz')
    factual=arrays(f);np.testing.assert_array_equal(factual['factual'],X[position]);free=list(map(int,pre['actionable_indices']))
    assert set(free).isdisjoint(factual['immutable']) and len(free)+len(factual['immutable'])==len(X[position])
    points=[];orpm=[]
    for n in cfg['model_counts']:
     for B in cfg['budgets'][ds]:
      directory=person/'neural'/f'models_{n}'/f'budget_{B:g}'/'ORPM'
      result=read(directory/'result.json');e=arrays(directory/'evaluation.npz')
      for name in ['result.json','evaluation.npz']:hashes[str(directory/name)]=sha(directory/name)
      np.testing.assert_array_equal(e['factual'],factual['factual'])
      calc=logits([nets[j] for j in indices(n)],e['point']);np.testing.assert_allclose(calc,e['logits'],rtol=2e-8,atol=1e-8)
      assert bool((calc>=0).all())==result['joint_validity']
      assert feasible(e['point'],factual['factual'],B,free,factual['lower'],factual['upper'])
      points.append(e['point']);orpm.append(dict(models=n,budget=float(B),joint_validity=result['joint_validity'],average_validity=result['average_validity'],cost=result['cost'],source_artifact=result['artifact']))
    np.savez_compressed(dest/'saved_orpm_points.npz',points=np.stack(points));save(dest/'saved_orpm.json',orpm)
    c=dict(key=key,dataset=ds,seed=seed,instance_id=iid,source_position=position,group=str(labs[position]),group_weight=counts[str(labs[position])]/len(X),bank=str(bdest.relative_to(root)),free=free,maximum_budget=float(max(cfg['budgets'][ds])))
    cases.append(c)
   print(utc(),'PREPARED_SAVED_BLOCK',ds,seed,flush=True)
 assert len(cases)==36 and len(full_cases)==1200
 # Interleave datasets, so early timing covers every dataset.
 ordered=[]
 for seed in cfg['pilot_seeds']:
  for slot in [0,1]:
   for ds in DATASETS:ordered.append([c for c in cases if c['seed']==seed and c['dataset']==ds][slot])
 save(root/'cohort_plan.json',ordered);save(root/'model_inventory.json',inventory);save(root/'full_cohort.json',full_cases)
 save(root/'group_counts.json',group_counts);save(root/'source_compute_demand.json',source_demand)
 look={(r['dataset'],r['seed'],r['instance_id']):r for r in full_cases}
 with open(TRADE/'results.csv',newline='') as f,open(root/'saved_orpm_results.csv','w',newline='') as out:
  reader=csv.DictReader(f);writer=csv.DictWriter(out,fieldnames=reader.fieldnames+['group','group_weight']);writer.writeheader();count=0
  for row in reader:
   if row['family']!='neural' or row['method']!='ORPM':continue
   c=look[(row['dataset'],int(row['seed']),int(row['instance_id']))];row.update(group=c['group'],group_weight=c['group_weight']);writer.writerow(row);count+=1
 assert count==26400
 save(root/'input_sha256.json',hashes)
 text='''Neural cost and disparity of multiplicity — timed reuse pilot 1.0

Purpose: extend the linear decomposition using the actual saved neural-only trade-off models and ORPM actions. No training and no new unseen-model evaluation. A full expansion is not automatically launched.

Source: the verified 10-seed x 20-person trade-off study, six datasets, nested 6/12/24-model sets and original L2 budget grids. These sets contain equal numbers of 20-ReLU and 50/100-ReLU networks at every size. They are different from the later one-architecture 6/12 study. Size comparisons keep architecture proportions fixed but also add different fitted models.

Reuse: all 26,400 neural ORPM setting outcomes for descriptive success curves and group coverage. For new minimum-cost computations select seeds 401-403 and two saved people per dataset/seed: first saved person from each protected group when both appear, otherwise first two; Synthetic and Polish first two. All 36 cases and model snapshots freeze before new optimization. No new cohort members.

New work: 864 individual-model minimum-L2 tasks plus 108 common-action minimum-L2 tasks, 972 planned tasks total. Already-accepted factuals have analytic zero cost. Each nontrivial MIQCP gets 10 seconds and one thread. The maximum sum of optimization allowances is 2.7 hours; construction, export, checks and reports are additional. This is a timing and implementation pilot, not a powered fairness study. Full saved-cohort expansion would contain 32,400 planned tasks before analytic zero-cost skips.

Action region: saved standardized coordinates, immutable features and bounds, within the largest saved dataset budget tau. Nominal acceptance is logit >= 0, matching saved trade-off results. Capped minimum cost is min(c,tau); failure to find a witness is unresolved, and a proof of no solution inside tau is not global infeasibility. Costs are standardized L2 distances, not money.

Encoding: both ReLU architectures represented exactly with interval-derived finite big-M bounds. Global bounds are numerical Gurobi bounds, not independently replayed branch-and-bound proofs. Native points are independently checked using saved network weights. Uncompressed MPS export followed by Python gzip avoids the earlier native compressed-export crash. Fixed algorithmic allowances are not equal wall-clock compute.

Saved ORPM points may tighten feasible cost upper bounds after solving, and may prove success at their saved budget; they never warm-start the optimizer. Thus these are structural cost bounds using all audited witnesses, not a new claim that the MIQCP algorithm found those ORPM points. Native incumbent availability, timeout status and bound widths are reported separately. A successful ORPM action provides an upper bound on the common minimum cost, not its optimum.

Success decomposition at each original budget: average single-model feasibility; every model separately feasible (possibly different actions); one common feasible action; saved ORPM joint success. Report lower/upper identification bounds whenever the minimum-cost interval straddles the budget. Separate the harder-model penalty, common-action penalty and ORPM shortfall. Enforce structural ordering only when mathematically justified. Do not count unresolved cases as failures.

Cost decomposition: common capped cost minus mean capped singleton cost, and common capped cost minus maximum capped singleton cost. Report interval subtraction with lower(A)-upper(B) and upper(A)-lower(B), respecting known nonnegativity. Do not infer minimum cost from ORPM's returned distance and do not average costs only among different subsets of successes.

Disparity: COMPAS recorded sex, ACSIncome recorded sex, German age <40 versus >=40, GMC age <40 versus >=40; these attributes remain immutable. No sensitive group claims for Synthetic or Polish. Compare within-group singleton burden, common-action burden and added multiplicity burden, then first-minus-second group gaps. Use only paired seeds with both groups, and report missing groups and sample counts. Pilot overall averages use each group's proportion in that seed's saved twenty-person cohort; group summaries remain conditional. Reused full-cohort means use the original twenty people and equal seed weights.

Uncertainty: pointwise descriptive 95% Student-t intervals across seed means: three-seed pilot df=2, ten-seed source summaries df=9. Show numerical identification ranges separately from sampling intervals, with conservative endpoint envelopes where appropriate. No independent-replication claim: all pilot subjects and original outcomes overlap the source. No causal or population fairness claim. Architecture/split/cohort/eligibility remain fixed. Every figure panel is saved separately as PNG/PDF. All inferential tables include interval endpoints and seed counts.

Completion: all 972 task outcomes, independent source/hash/point/bound/encoding checks, cost and disparity reports and archives, timing-based expansion estimate, final completion log entry and successful supervised exit. Timeouts are reportable outcomes; invalid bounds or inconsistent forward predictions are integrity failures. The supervisor records Python exceptions and native crashes. No full-study certification-rate gate.
'''
 for p in [root,droot]:
  (p/'PROTOCOL.txt').write_text(text);(p/'PROTOCOL.html').write_text('<!doctype html><meta charset="utf-8"><title>Neural multiplicity pilot protocol</title><style>body{max-width:1000px;margin:3em auto;font:17px/1.6 system-ui;padding:0 2em}pre{white-space:pre-wrap}</style><pre>'+html.escape(text)+'</pre>')
  save(p/'shared_run.json',dict(computation=str(root),cost_report=str(root/'cost_report'),disparity_report=str(droot)))
 frozen={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
 save(root/'freeze.json',dict(utc=utc(),before_new_optimization=True,sha256=frozen))
 save(root/'preparation_verification.json',dict(passed=True,cases=36,source_cases=1200,source_orpm_rows=26400,pilot_settings=sum(len(cfg['budgets'][c['dataset']])*3 for c in cases),all_pilot_orpm_forward_and_feasibility_checks_passed=True,no_model_training=True))
 save(root/'progress.json',dict(status='prepared',completed_tasks=0,expected_tasks=972,updated_utc=utc()))
 return cfg
