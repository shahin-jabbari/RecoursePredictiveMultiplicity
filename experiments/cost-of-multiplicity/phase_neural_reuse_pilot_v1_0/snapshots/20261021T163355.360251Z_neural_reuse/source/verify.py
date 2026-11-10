import gzip,hashlib,json
from pathlib import Path
import numpy as np
from common import *

def verify(root,check_reports=False):
 root=Path(root);cfg=read(root/'protocol.json');count=0;primal=0;encoded=0
 for p,h in read(root/'input_sha256.json').items():assert sha(p)==h,'Source modified: '+p
 for p,h in read(root/'freeze.json')['sha256'].items():assert sha(root/p)==h,'Frozen pilot input modified: '+p
 for c in read(root/'cohort_plan.json'):
  folder=root/'cases'/c['key'];f=arrays(folder/'factual.npz');x0=f['factual'];nets={j:load_net(root/c['bank']/f'model_{j:04d}.npz') for j in range(24,48)}
  files=list((folder/'tasks').glob('*/result.json'));assert len(files)==27
  original_points=arrays(folder/'saved_orpm_points.npz')['points']
  for j,net in nets.items():np.testing.assert_allclose(forward(net,original_points),torch_forward(root/c['bank']/f'model_{j:04d}.npz',original_points),rtol=1e-10,atol=1e-9)
  for p in files:
   r=read(p);info=r['info'];point=arrays(p.with_name('point.npz'))['point'];js=r['indices'];count+=1
   assert info['capped_lower']>=0 and info['capped_upper']<=c['maximum_budget']+1e-8 and info['capped_lower']<=info['capped_upper']+1e-5
   if info['available']:
    assert feasible(point,x0,c['maximum_budget'],c['free'],f['lower'],f['upper'])
    z=np.array([torch_forward(root/c['bank']/f'model_{j:04d}.npz',point) for j in js])
    # Independent float64 operation ordering can differ at machine precision.
    assert z.min()>=-1e-10 and logits([nets[j] for j in js],point).min()>=0
    assert abs(float(np.linalg.norm(point-x0))-info['cost_upper'])<1e-8
    primal+=1
   if info.get('raw_point') is not None:
    raw=np.array(info['raw_point']);actual=logits([nets[j] for j in js],raw)
    independent=np.array([torch_forward(root/c['bank']/f'model_{j:04d}.npz',raw) for j in js]);np.testing.assert_allclose(actual,independent,atol=1e-9,rtol=1e-10);encoded+=1
   if info.get('export_uncompressed_sha256'):
    h=hashlib.sha256()
    with gzip.open(p.parent/'problem.mps.gz','rb') as z:
     for chunk in iter(lambda:z.read(1<<20),b''):h.update(chunk)
    assert h.hexdigest()==info['export_uncompressed_sha256']
   if info['infeasible_within_horizon']:assert not info['available']
 assert count==cfg['expected_tasks']
 report_checks=None
 if check_reports:
  import zipfile,pandas as pd
  from analysis import pilot_rows
  costs,success,_=pilot_rows(root);assert len(costs)==540 and len(success)==sum(len(cfg['budgets'][c['dataset']])*21 for c in read(root/'cohort_plan.json'))
  for destination,number in [(root/'cost_report',30),(Path(cfg['disparity_result']),16)]:
   assert len(list((destination/'figures').glob('*.png')))==number
   assert len(list((destination/'figures').glob('*.pdf')))==number
   for p in (destination/'tables').glob('*95ci.csv'):
    df=pd.read_csv(p);assert 'seeds' in df and 'lower_endpoint_ci95_low' in df and len(df)>0
    assert (df.estimate_lower<=df.estimate_upper+1e-6).all()
   with zipfile.ZipFile(destination/'figures_tables.zip') as z:assert z.testzip() is None
   assert (destination/'report.txt').exists() and (destination/'report.html').exists()
  report_checks=True
 return dict(passed=True,tasks=count,independently_validated_witnesses=primal,raw_incumbent_forward_checks=encoded,source_preserved=True,frozen_inputs_preserved=True,report_checks=report_checks,utc=utc(),scope='Exact ReLU encoding validated on analytic and fixed-input tests; global bounds remain numerical solver certificates, not a replayed search proof.')
