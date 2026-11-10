from pathlib import Path
import json
from collections import defaultdict
# Original measured aggregate seconds for only ORPM + exact margin, per setting.
SETTING={'COMPAS':.10407*3600/4200,'Folktables':.11296*3600/4800,'German':.29979*3600/2814,'Give_Me_Some_Credit':.24215*3600/2376}
TOTAL={'COMPAS':4200,'Folktables':4800,'German':2814,'Give_Me_Some_Credit':2376}
def remaining(root):
    root=Path(root);timings=[]
    if (root/'timing.jsonl').exists():
        for l in (root/'timing.jsonl').read_text().splitlines():
            try:timings.append(json.loads(l))
            except json.JSONDecodeError:pass
    samples=defaultdict(list);used=0
    for r in timings:
        if r['kind'] in ['setting_wall','min_l2']:used+=r['seconds']
        if r['kind']=='setting_wall':samples[r['dataset']].append(r['seconds'])
    central=0;breakdown={}
    for ds,total in TOTAL.items():
        vals=samples[ds];mean=sum(vals)/len(vals) if len(vals)>=30 else SETTING[ds]
        sec=max(0,total-len(vals))*mean
        breakdown[ds]=dict(done=len(vals),total=total,seconds_per_setting=mean,remaining_hours=sec/3600)
        central+=sec
    # Shared minimum-QP times are tiny; initial rate includes amortized reference
    # minimum-QP work. A small reserve plus verification/reporting is explicit.
    central=central/3600+.6
    return dict(remaining_hours_central=central,planning_hours_lower=max(.25,central*.75),planning_hours_upper=max(1.5,central*1.8),timed_new_work_hours=used/3600,breakdown=breakdown,note='Elapsed setting wall time includes ORPM, margin optimization and point/record writes. Min-L2 cached once. 0.6h reserve for shared work, verification, reports; sleep not predicted.')
if __name__=='__main__':
    import sys
    print(json.dumps(remaining(Path(sys.argv[1])),indent=2))
