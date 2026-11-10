from pathlib import Path
p=Path(__file__).parent;source=Path('/Users/sj987/Documents/ChatGPT/Code/Predictive_Multiplicity_Recourse/V7/experiments/linear_exact_unseen/results/20261018T180330.097982Z_98de9f6e9a/source/analyze.py').read_text()
source=source[:source.index('    def value(ds,m,metric,percent=False):')]
source=source.replace("rows=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()];df=pd.DataFrame([{k:v for k,v in r.items() if k!='info'} for r in rows]);", "df=pd.DataFrame([{k:v for k,v in json.loads(s).items() if k!='info'} for s in open(root/'records.jsonl')]);")
source=source.replace(' · pilot',' · full study').replace('Pointwise 95% t intervals across 3 seeds; 5 people/seed. Rate bands clipped to [0,1].','Pointwise 95% t CIs: 10 seeds, 20 people/seed. Rate bands clipped to [0,1].')
source=source.replace("if metric.endswith('validity'):ax.set_ylim(-.03,1.03)", "if metric.endswith('validity'):\n                    ax.set_ylim(-.03,1.03);ax.set_yticks(np.linspace(0,1,6));ax.set_yticklabels([f'{v:.1f}' for v in np.linspace(0,1,6)]);ax.tick_params(labelleft=True)",1)
# Second occurrence has one less indentation level.
source=source.replace("if metric.endswith('validity'):ax.set_ylim(-.03,1.03)", "if metric.endswith('validity'):\n                ax.set_ylim(-.03,1.03);ax.set_yticks(np.linspace(0,1,6));ax.set_yticklabels([f'{v:.1f}' for v in np.linspace(0,1,6)]);ax.tick_params(labelleft=True)")
(p/'analyze.py').write_text(source)
