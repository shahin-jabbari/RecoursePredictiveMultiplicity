from pathlib import Path
import json,datetime,statistics,collections
W=Path(__file__).resolve().parent;P=json.loads((W/'protocol.json').read_text());S={k:Path(v) for k,v in P['source_runs'].items()};C,T,K,U=(S[k] for k in ['convergence','tradeoff','multiplicity','unseen'])
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
cal=collections.defaultdict(list)
for z in rows(C/'summaries.jsonl'):cal[(z['dataset'],z['models'],z['budget'])].append(z)
tr=collections.defaultdict(list)
for z in rows(T/'timing.jsonl'):
 if z['kind']=='method':tr[(z['dataset'],z['models'],z['budget'],z['method'])].append(z['seconds'])
extra={(z['dataset'],z['seed'],z['position']):z for z in read(K/'cohort_plan.json') if not z['preserved']}
cohorts=[];outside=[];prefix=refs=0;ctime=0;ttime=collections.defaultdict(float);nnew=collections.Counter()
for ds in P['datasets']:
 for seed in P['seeds']:
  meta=read(C/'banks'/ds/f'seed_{seed}/bank.json');standard=meta['eligible_random_order'][:20];assert len(standard)==20
  old=[z for z in read(K/'cohort_plan.json') if z['dataset']==ds and z['seed']==seed];costpos=[z['position'] for z in old]
  for pos in meta['eligible_random_order']:
   if len(costpos)==20:break
   if pos not in costpos:costpos.append(pos)
  assert len(costpos)==20 and len(set(costpos))==20
  outsiders=[z for z in old if z['position'] not in standard];outside+=outsiders
  cohorts.append(dict(dataset=ds,seed=seed,standard_positions=standard,cost_disparity_positions=costpos,retained_prior_cost_cases=len(old),cost_retained_outside_standard=len(outsiders)))
  for person,pos in enumerate(standard):
   for m in P['model_counts']:
    for b in P['positive_budgets'][ds]:
     if person<3:continue
     pref=person<5 or (ds,seed,pos) in extra;ref=person<5;prefix+=int(pref);refs+=int(ref)
     ctime+=statistics.mean(v['algorithm_wall_seconds']*(.5 if pref else 1)+v['diagnostic_wall_seconds']+(0 if ref else v['reference_wall_seconds']) for v in cal[(ds,m,b)])
   budgets=read(T/'protocol.json')['budgets'][ds]
   for m in P['model_counts']:
    for b in budgets:
     if b==0:continue
     for method in ['ORPM','UniformTree','DirectMinimax']:
      reusable=person<5 or (method in ['ORPM','DirectMinimax'] and b in P['positive_budgets'][ds]) or (method=='ORPM' and (ds,seed,pos) in extra)
      if reusable:continue
      samples=tr[(ds,m,b,method)];assert samples
      ttime[method]+=statistics.mean(samples);nnew[method]+=1
assert prefix==1629 and refs==1080
f=read(U/'random_forest_pilot/full_study_estimate.json');forest_methods=sum(f['methods_hours'].values())*4-f['measured_pilot_method_seconds']/3600
forest_preparation=f['preparation_hours']*.8;forest_vr=f['verification_reporting_hours']*4
forest=forest_methods+forest_preparation+forest_vr
conv=ctime/3600+1.5;trade=sum(ttime.values())/3600+2;cost=2.;unseen=3.
stages=[dict(id='convergence',title='Decision-tree convergence',central_hours=conv,planning_hours=[ctime/3600*.65+.5,ctime/3600*1.8+3],total_cases=1200,total_trajectories=10800,total_rounds=2160000,preserved_complete_trajectories=1620,preserved_prefixes=1629,new_response_calls=1673100,new_checkpoint_calls=137700,new_direct_references=8100,report='/Users/sj987/Documents/ChatGPT/Recourse/output/Tree-Convergence-Report-20261023/main.tex'),
 dict(id='tradeoff',title='Decision-tree trade-offs',central_hours=trade,planning_hours=[sum(ttime.values())/3600*.65+1,sum(ttime.values())/3600*1.8+4],total_cases=1200,total_cells=25200,total_method_rows=100800,new_tasks=dict(nnew),new_method_hours={k:v/3600 for k,v in ttime.items()},shared_reference_enumerations=5400,report='/Users/sj987/Documents/ChatGPT/Recourse/output/Tree-Tradeoff-Report-20261024/main.tex'),
 dict(id='multiplicity',title='Decision-tree cost and disparity',central_hours=cost,planning_hours=[1,6],total_cases=1200,preserved_cases=372,new_cases=828,total_singletons=28800,new_singletons=19872,total_joint_tasks=3600,new_joint_tasks=2484,total_ORPM_settings=25200,new_ORPM_optimizations=0,report='/Users/sj987/Documents/ChatGPT/Recourse/output/pdf/Tree-Multiplicity-Report-20261024/main.tex'),
 dict(id='decision_tree_unseen',title='Decision-tree unseen evaluation',central_hours=unseen,planning_hours=[1,6],total_cases=1200,total_method_rows=100800,total_panel_evaluations=201600,new_panel_evaluations=151200,reused_unseen_trees=6000,new_training=0,new_recourse_optimization=0,report='/Users/sj987/Documents/ChatGPT/Recourse/output/pdf/Tree-Unseen-Generalization-Report-20261024/main.tex'),
 dict(id='random_forest',title='Same-architecture random-forest full study',central_hours=forest,planning_hours=[forest_methods*.5+forest_preparation+forest_vr*.5,forest_methods*2.5+forest_preparation*2+forest_vr*2],total_cases=1200,positive_cells=10800,zero_cells=3600,total_method_rows=57600,total_panel_evaluations=115200,total_response_calls=1080000,preserved_response_calls=4800,new_response_calls=1075200,preserved_banks=12,new_banks=48,whole_forests_per_set=[6,12,24],trees_per_forest=10,depth=4,report='/Users/sj987/Documents/ChatGPT/Recourse/output/pdf/Tree-Unseen-Generalization-Report-20261024/main.tex')]
plan=dict(version='1.0',authorization='Expand all existing tree studies to ten seeds and twenty instances per dataset/seed; sequentially finish, verify and update the concise report, then advance. Updates every two hours.',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hard_deadline=None,scope='Existing decision-tree convergence/tradeoff/cost-disparity/unseen studies plus expansion of same-architecture forest pilot. No additional forest cost/fairness/convergence study.',sources={k:str(v) for k,v in S.items()},stages=stages,cohort_policy='Standard studies use first twenty saved eligible people per bank. Cost/disparity preserves all 372 prior cases and fills each bank to twenty in saved eligible order; 11 prior cases lie outside standard prefix. Report both cohort provenance and conditional demographic support.',cohort_overlap=1189,retained_nonstandard_cases=outside,cohorts=cohorts,central_total_hours=sum(z['central_hours'] for z in stages),planning_total_hours=[sum(z['planning_hours'][i] for z in stages) for i in [0,1]],timing_caveats=['Awake compute projection, not a guarantee or confidence interval.','Forest rates calibrated on two seeds and one person; twelve-model and middle-budget cells interpolated.','No double counting nested preparation/panel timings, shared references, or historical solve times.','Before each later stage recompute task reuse and ETA from verified expanded preceding outputs.','Integrity gates are independent of favorable efficacy or certification.','The total includes stage report reserves; a two-hour review cadence can add small interstage delays.'],report_policy='Back up existing concise report before updating it in place. Three paragraphs: setting/methodology and two results paragraphs; main selected figures, remaining appendix; PDF, standalone figures and source bundle. Pointwise 95% seed-t CIs, people within seed then equal seed weights. Keep budgets/counts separate. Compilation and visual inspection required. Never describe incomplete/unverified results as finished.')
(W/'campaign_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({k:plan[k] for k in ['central_total_hours','planning_total_hours']},indent=2))
for z in stages:print(z['id'],round(z['central_hours'],2),[round(x,2) for x in z['planning_hours']],z.get('new_tasks'))
