from experiment import *

def full_estimate(root):
 p=read(root/'protocol.json');timing=rows(root/'timing.jsonl');assert p['arm']=='random_forest_pilot'
 measured=[r for r in timing if r['kind']=='method'];refs=[r for r in timing if r['kind']=='reference_shared'];prep=[r for r in timing if r['kind']=='preparation_bank']
 details=[];totals=defaultdict(float)
 for ds in p['datasets']:
  lo,mid,hi=p['full_positive_budgets'][ds]
  for method in ['ORPM','UniformForest','DirectMinimax']:
   for m in [6,12,24]:
    for b in [lo,mid,hi]:
     sources=[];rate=0.
     for mm,wm in ([(m,1.)] if m in [6,24] else [(6,2/3),(24,1/3)]):
      bs=[(b,1.)] if b in [lo,hi] else [(lo,1-np.log(b/lo)/np.log(hi/lo)),(hi,np.log(b/lo)/np.log(hi/lo))]
      for bb,wb in bs:
       q=[r['seconds'] for r in measured if r['dataset']==ds and r['method']==method and r['models']==mm and r['budget']==bb]
       assert len(q)==2,(ds,method,mm,bb,len(q))
       rate+=wm*wb*np.mean(q);sources.append(dict(models=mm,budget=bb,times=q,weight=wm*wb))
     seconds=50*rate;totals[method]+=seconds
     details.append(dict(dataset=ds,method=method,models=m,budget=b,cases=50,seconds=seconds,interpolated=m==12 or b==mid,calibration=sources))
  for b in [lo,mid,hi]:
   if b in [lo,hi]:q=[r['seconds'] for r in refs if r['dataset']==ds and r['budget']==b];rate=float(np.mean(q))
   else:
    weight=np.log(b/lo)/np.log(hi/lo);rate=(1-weight)*np.mean([r['seconds'] for r in refs if r['dataset']==ds and r['budget']==lo])+weight*np.mean([r['seconds'] for r in refs if r['dataset']==ds and r['budget']==hi])
   totals['SingleForestReference_shared']+=50*rate
 solve=sum(totals.values());spent=sum(r['seconds'] for r in measured+refs)
 preparation=sum(r['seconds'] for r in prep)*5
 verification=read(root/'verification.json')['seconds'] if (root/'verification.json').exists() else 600
 evaluations=sum(r['seconds'] for r in timing if r['kind']=='evaluation_bank')*50
 # Verification includes bank checks and method checks. Scaling all by 50 is deliberately conservative for the bank component.
 post=50*verification+evaluations+1800
 central=(solve+preparation+post)/3600
 additional=(solve-spent+preparation-sum(r['seconds'] for r in prep)+post)/3600
 return dict(utc=now(),scope='10 seeds x 5 saved people x 6 datasets x 3 counts x 3 positive budgets, plus zero controls',full_launch_authorized=False,expected_positive_cells=2700,expected_zero_cells=900,expected_method_rows=14400,expected_response_calls=270000,methods_hours={k:v/3600 for k,v in totals.items()},preparation_hours=preparation/3600,verification_reporting_hours=post/3600,central_total_hours=central,central_additional_hours=additional,planning_additional_hours=[max(0,(solve-spent)*.5/3600+(preparation-sum(r['seconds'] for r in prep))*.7/3600+post*.5/3600),((solve-spent)*2.5+(preparation-sum(r['seconds'] for r in prep))*2+post*2)/3600],measured_pilot_method_seconds=spent,calibration_seeds=[401,406],pilot_people_per_seed=1,interpolation='linear in model count between 6 and 24; linear in log budget between low and high; medium budget and m12 unmeasured',uncertainty='Planning scenario range, not a CI or guarantee; only two seeds and one person per seed; intermediate budgets can be slower than endpoints; excludes future implementation changes',optimization_cap_sum_hours=127.5,cap_sum_excludes='encoding, training, serialization, verification, reports; not an elapsed-time bound',matched_rates=details)

def remaining(root):
 root=Path(root);p=read(root/'protocol.json');pr=read(root/'progress.json') if (root/'progress.json').exists() else {};timing=rows(root/'timing.jsonl')
 if pr.get('status')=='completed':return dict(utc=now(),remaining_hours=0,completed=True)
 n=len([r for r in timing if r['kind']=='preparation_bank']);target=60 if p['arm']=='decision_tree' else 12
 if not (root/'freeze.json').exists():
  if n:
   rates=[r['seconds'] for r in timing if r['kind']=='preparation_bank'];training=(target-n)*np.mean(rates)/3600
  else:training=.5 if p['arm']=='decision_tree' else .75
  reserve=.5 if p['arm']=='decision_tree' else 2.
  return dict(utc=now(),remaining_hours=training+reserve,planning_hours=[max(.1,(training+reserve)*.5),(training+reserve)*2.5],phase='preparation',measured_banks=n,warning='Unmeasured dataset/panel strata and all future optimization remain provisional')
 if p['arm']=='decision_tree':return dict(utc=now(),remaining_hours=.5,planning_hours=[.1,1.5],phase=pr.get('status'),warning='Evaluation/verification/reporting reserve, not fitted solver time')
 completed=[r for r in timing if r['kind']=='method'];seconds=0.;fallbacks=0
 for c in read(root/'task_plan.json')['cells']:
  if c['budget']==0:continue
  for method in ['ORPM','UniformForest','DirectMinimax']:
   if any(r['key']==c['key'] and r['method']==method for r in completed):continue
   candidates=[r['seconds'] for r in completed if r['dataset']==c['dataset'] and r['models']==c['models'] and r['budget']==c['budget'] and r['method']==method]
   if not candidates:candidates=[r['seconds'] for r in completed if r['method']==method];fallbacks+=1
   seconds+=float(np.mean(candidates)) if candidates else (100 if method=='ORPM' else 30)
 refdone=len([r for r in timing if r['kind']=='reference_shared']);seconds+=max(0,24-refdone)*30
 return dict(utc=now(),remaining_hours=seconds/3600+.5,planning_hours=[seconds/3600*.5+.15,seconds/3600*2.5+1],unmatched_strata=fallbacks,phase=pr.get('status'),note='New completed wall times include optimization; shared references counted once. Verification/reporting reserve included; planning range is not a CI.')

if __name__=='__main__':print(json.dumps(clean(remaining(Path(sys.argv[1]))),indent=2))
