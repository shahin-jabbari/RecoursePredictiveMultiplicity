"""Presentation-only review; does not alter protocol, interventions or estimators."""
from pathlib import Path
import sys,json,hashlib,html,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(sys.argv[1])/'source'))
from common import read,save,sha,utc
from analyze import COLORS,LABELS
root=Path(sys.argv[1]);
import shutil
shutil.copy2(Path(__file__),root/'presentation_review.py')
cfg=read(root/'protocol.json');cell=pd.read_csv(root/'tables/all_cells_95ci.csv');before={str(p.relative_to(root)):sha(p) for p in [root/'records.jsonl',root/'verification.json']+list((root/'tables').glob('*'))}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','font.family':'DejaVu Sans','pdf.fonttype':42})
for ds in cfg['datasets']:
    short='GMC' if ds=='Give_Me_Some_Credit' else ds
    for metric in ['seen_joint_validity','unseen_joint_validity','seen_mean_validity','unseen_mean_validity','exact_loss_gap']:
        part=cell[(cell.dataset==ds)&(cell.budget==max(cfg['budgets'][ds]))&(cell.metric==metric)];fig,ax=plt.subplots(figsize=(7.6,4.9))
        for k,m in enumerate(cfg['methods']):
            q=part[part.method==m].sort_values('model_count');xx=q.model_count.to_numpy();yy=q['mean'].to_numpy();ax.plot(xx,yy,marker='o',ms=3,color=COLORS[k],label=m,lw=1.6,ls='--' if m.startswith('Exact') else '-');ax.fill_between(xx,q.plot_low.to_numpy(),q.plot_high.to_numpy(),color=COLORS[k],alpha=.10)
        ax.set(xlabel='Number of optimization models (nested prefixes)',ylabel=LABELS[metric],title=f'{short} · budget {max(cfg["budgets"][ds]):g} · pilot');ax.set_xticks(cfg['model_sizes'])
        if metric.endswith('validity'):
            ax.set_ylim(-.03,1.03);ax.set_yticks(np.linspace(0,1,6));ax.set_yticklabels([f'{x:.1f}' for x in np.linspace(0,1,6)])
        if metric=='exact_loss_gap':ax.set_yscale('symlog',linthresh=.01)
        ax.tick_params(axis='both',labelleft=True,labelbottom=True);ax.grid(alpha=.2);ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.17),fontsize=8,frameon=False);fig.text(.01,.01,'Pointwise 95% t intervals across 3 seeds; 5 people/seed. Rate bands clipped to [0,1].',fontsize=8);fig.subplots_adjust(bottom=.26);name=f'{short}_{metric}_model_count';fig.savefig(root/'figures'/(name+'.png'),dpi=150);fig.savefig(root/'figures'/(name+'.pdf'));plt.close(fig)
notes='''READINESS AND PRIMARY INTERPRETATION — POST-RUN REVIEW

The pipeline is ready for the proposed larger linear study within its frozen
scope. All 3,780 records and 675 exact certificates passed independent checks.
The full study has not been launched.

At 24 optimization models and the largest pilot budget, ORPM and ExactMaxMargin
have identical seen and unseen joint-validity outcomes on every sampled person.
ORPM mean worst-logistic-loss gaps above the exact solution, with 95% seed-t CIs:
Synthetic 0.000021 [-0.000001, 0.000043]; COMPAS 0.004407 [-0.000314, 0.009127];
GMC 0.000387 [-0.000162, 0.000936]. Negative interval limits reflect the t
approximation; the population excess loss is nonnegative up to solver tolerance.

GMC is the clearest scientific warning. At budget 75, ORPM and ExactMaxMargin
both have 100% [100%,100%] seen joint validity but 0% [0%,0%] unseen joint validity.
ElliCE has 100% [100%,100%] on both panels. ORPM still averages 88.8% acceptance
across the unseen models (95% raw seed-t CI 73.0%–104.6%). Thus high average
acceptance can coexist with failure of the all-50 joint criterion. The tiny
seen-objective gap suggests that merely increasing ORPM iterations will not
resolve this mismatch between finite-bank optimization and the unseen panel.
These degenerate 0/100% intervals mean identical seed means, NOT population
certainty; only three seeds and fifteen individuals per dataset were sampled.

The unseen hypotheses were drawn from a training-Hessian ellipsoid around the
reference logistic model. This is aligned with ElliCE's uncertainty family,
although admission checks actual training loss and validation accuracy and the
Hessian regularization differs. It is a transparent conditional stress test,
not distribution-neutral evidence that one method generalizes universally.
A broader generalization claim will require separately frozen panels spanning
unseen feature masks and independently retrained or different-family models.
Such extra panels were not added after seeing these results.

The largest-budget 6-to-24-model contrast does not change ORPM joint validity
on either panel in this pilot (paired difference 0 [0,0] percentage points for
each dataset). GMC unseen mean acceptance increases by 5.73 pp (95% CI -14.00
to 25.47 pp), an imprecise trend. The full budget/count tables and separate
figures retain all settings, including zero-budget controls.

Exact minimum-cost interventions sacrifice seen margin and can transfer worse.
For example, Synthetic unseen joint validity falls to 60.0% (raw 95% seed-t CI
-39.4%–159.4%) for ExactMinL2 versus 100% [100%,100%] for ExactMaxMargin at budget 3.
No superiority claim is supported by these wide intervals.

Next study: preserve this protocol, cohorts/panels and pilot outputs while
expanding to six datasets, ten seeds and twenty individuals. Treat the three
untested datasets' admission/cohort checks as preconditions. Use the larger
study to estimate the optimization/transfer trade-off, not to assume dominance.
Exact comparisons here are QP/SOCP in the continuous L2 action space; a literal
MILP/discrete-action comparison is a different protocol.
'''
(root/'INTERPRETATION.txt').write_text(notes)
report=(root/'report.txt').read_text();report=notes+'\n\n'+report
report+='\nAvailability note: exact_loss_gap averages only available interventions; it is undefined for no-output placeholders. Primary validity always retains all selected people.\n'
(root/'report.txt').write_text(report)
page=(root/'report.html').read_text();a=page.index('<pre>')+len('<pre>');b=page.index('</pre>',a);page=page[:a]+html.escape(report)+page[b:];(root/'report.html').write_text(page)
assert all(sha(root/p)==h for p,h in before.items())
save(root/'presentation_review.json',dict(reviewed_utc=utc(),changes=['explicit model-count y-axis labels','descriptive interpretation added'],unchanged_results_and_tables_sha256=before,script_sha256=sha(Path(__file__)),statistical_estimators_unchanged=True,protocol_unchanged=True))
with zipfile.ZipFile(root/'figures_tables.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(list((root/'figures').glob('*'))+list((root/'tables').glob('*'))+[root/n for n in ['report.txt','report.html','protocol.json','PROTOCOL.txt','verification.json','analysis.json','INTERPRETATION.txt','presentation_review.json','presentation_review.py']]):z.write(p,p.relative_to(root))
with zipfile.ZipFile(root/'figures_tables.zip') as z:assert z.testzip() is None
print('Presentation reviewed; intervention results, protocol and CI tables unchanged.')
