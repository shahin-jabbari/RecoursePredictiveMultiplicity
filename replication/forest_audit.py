"""Build a new contradiction ledger from a fresh forest replication."""
from collections import defaultdict
import numpy as np
from io_tools import append

def audit_cell(root,cell,records,factual):
    from common import method_dir,read,sha
    from certificate_audit import scan_cell,prove_group
    rejected,rounds,names,points,losses,_=scan_cell(root,cell,records,factual)
    groups=defaultdict(list)
    for key,method,iteration in sorted(rejected):
        folder=method_dir(root,cell,method)
        row=records[method]
        v=dict(**cell,method=method,source_person_id=row['source_person_id'],
            original_outcome_sha256=sha(folder/'outcome.json'),record_id=row['record_id'])
        if method=='ORPM':
            info=rounds[iteration-1]['oracle']
            w=np.asarray(info['weights'])
            values=losses@w
            v.update(round=iteration,weights=info['weights'],original_rounds_sha256=sha(folder/'rounds.jsonl'))
        else:
            info=read(folder/'solver_outcome.json')['oracle']
            values=losses.mean(axis=1) if method=='UniformForest' else losses.max(axis=1)
            v['original_solver_outcome_sha256']=sha(folder/'solver_outcome.json')
        wi=int(values.argmin())
        v.update(witness=names[wi],witness_point=points[wi].tolist(),lower_bound=info['lower_bound'])
        v['witness_weighted_loss' if method=='ORPM' else 'witness_objective']=float(values[wi])
        groups[method].append(v)
    for method,claims in groups.items():
        details,overlay=prove_group(root,cell,method,claims,records[method],rounds,names,points,losses)
        for detail in details:append(root/'certificate_rejections.jsonl',detail)
        append(root/'audited_status.jsonl',overlay)
    append(root/'certificate_screen.jsonl',dict(cell=cell['key'],audited_rejections=len(rejected)))
