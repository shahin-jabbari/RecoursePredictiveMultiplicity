"""Run all configured study grids into a new directory; never resume or overwrite."""
import argparse,subprocess,sys
from pathlib import Path
from run import ROOT,STUDIES
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",required=True,type=Path);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    for study in STUDIES:
        cmd=[sys.executable,"-B",str(ROOT/"run.py"),"run",study,"--out",str(out/study)]
        if study=="tree-literature":cmd.extend(["--orpm-from",str(out/"tree-tradeoff")])
        subprocess.run(cmd,check=True)
