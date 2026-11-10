"""Seed-level uncertainty and paired comparisons for repeated-holdout recourse.

Intervals are pointwise Student-t intervals, conditional on the dataset and protocol.
They are not simultaneous confidence bands, population-sampling guarantees, or
independent-instance intervals. Identical observed seed means are flagged, not
presented as evidence of zero uncertainty.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

CONFIDENCE = .95


def mean_ci(values, bounds=None, confidence=CONFIDENCE):
    """Summarize equally weighted seed means; no pseudo-replication of factuals."""
    if not 0 < confidence < 1:
        raise ValueError('confidence must lie strictly between zero and one')
    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    n = len(a)
    mean = float(a.mean()) if n else np.nan
    sd = float(a.std(ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    low = high = raw_low = raw_high = np.nan
    status = 'no_observations' if n == 0 else 'insufficient_seeds' if n == 1 else 'few_seeds' if n < 5 else 'ok'
    if n > 1:
        half = float(student_t.ppf((1 + confidence)/2, n-1)) * se
        raw_low, raw_high = mean-half, mean+half
        if sd == 0:
            status = 'zero_observed_seed_variance'
        else:
            low, high = raw_low, raw_high
            if bounds is not None:
                low, high = max(bounds[0], low), min(bounds[1], high)
    return dict(mean=mean, sd=sd, se=se, ci_low=low, ci_high=high,
                ci_low_unbounded=raw_low, ci_high_unbounded=raw_high,
                n_seeds=n, confidence=confidence, ci_status=status)


def seed_values(records, metric, scale=1.0):
    """Return one mean per seed; unavailable metrics never become zero."""
    frame = pd.DataFrame(records)
    if frame.empty or 'seed' not in frame or metric not in frame:
        return pd.Series(dtype=float)
    data = frame[['seed', metric]].copy()
    data[metric] = pd.to_numeric(data[metric], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return data.groupby('seed')[metric].mean().dropna() * scale


def metric_summary(records, metric, scale=1., bounds=None):
    return mean_ci(seed_values(records, metric, scale), bounds)


def columns(prefix, stats, unit=''):
    """Consistent numeric columns for summaries and plotting; SD is not CI."""
    suffix = f' ({unit})' if unit else ''
    result = {f'{prefix} {label}{suffix}': stats[key] for label, key in [
        ('Mean','mean'), ('Std','sd'), ('SE','se'),
        ('CI95 Low','ci_low'), ('CI95 High','ci_high'),
        ('CI95 Unbounded Low','ci_low_unbounded'), ('CI95 Unbounded High','ci_high_unbounded')]}
    result.update({f'{prefix} N Seeds': stats['n_seeds'], f'{prefix} CI Status': stats['ci_status']})
    return result


def format_ci(mean, low, high, status, digits=2):
    if not np.isfinite(mean):
        return 'N/A (no observations)'
    if not (np.isfinite(low) and np.isfinite(high)):
        return f'{mean:.{digits}f} (CI unavailable: {status})'
    caution = '; exploratory, <5 seeds' if status == 'few_seeds' else ''
    return f'{mean:.{digits}f} [95% CI {low:.{digits}f}, {high:.{digits}f}{caution}]'


def summarize_seed_frame(frame, group_columns, metrics, bounds=None):
    """The input already contains exactly one row per seed/group (e.g. round)."""
    rows = []
    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, 'metric', 'mean', 'sd', 'se', 'ci_low', 'ci_high', 'n_seeds', 'ci_status'])
    for key, group in frame.groupby(group_columns, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        if group.seed.duplicated().any():
            raise ValueError('Expected one value per seed and group')
        for metric in metrics:
            rows.append({**dict(zip(group_columns, key)), 'metric': metric,
                         **mean_ci(group[metric], (bounds or {}).get(metric))})
    return pd.DataFrame(rows)


def per_seed_table(records):
    frame = pd.DataFrame(records)
    if frame.empty or not {'seed','method','instance_id','valid'} <= set(frame):
        return pd.DataFrame()
    frame = frame[frame.instance_id.notna() & frame.valid.notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    frame['successful_cost'] = frame['cost'].where(frame.valid.astype(bool))
    group = [k for k in ['dataset','family','aggregation','budget','method','seed'] if k in frame]
    agg = {'instances':('valid','size'), 'validity_pct':('valid','mean')}
    for target, source in [('mean_seconds','elapsed_seconds'), ('mean_cost','cost'),
        ('mean_successful_cost','successful_cost'), ('mean_loss','max_loss'),
        ('unseen_validity_pct','unseen_all_valid'), ('unseen_fraction_pct','unseen_fraction_valid'),
        ('model_fraction_pct','model_fraction_valid')]:
        if source in frame:
            frame[source] = pd.to_numeric(frame[source], errors='coerce')
            agg[target] = (source,'mean')
    result = frame.groupby(group, dropna=False).agg(**agg).reset_index()
    for col in [c for c in result if c.endswith('_pct')]:
        result[col] *= 100
    return result


def paired_comparisons(records):
    """Pair by factual within seed, then equally weight seed-level differences.

    Each grouping is confined to one saved run and one dataset/family/budget.
    Generalization optimization/evaluation splits are compared separately.
    Missing pairs are reported and excluded, never counted as a tie or failure.
    """
    frame = pd.DataFrame(records)
    base_columns = ['reference','baseline','metric','paired_instances','completed_seeds',
                    'mean_difference','ci_low','ci_high','ci_status']
    if frame.empty or not {'seed','method','instance_id','valid'} <= set(frame):
        return pd.DataFrame(columns=base_columns), pd.DataFrame()
    frame = frame[frame.instance_id.notna() & frame.valid.notna()].copy()
    groups = [k for k in ['dataset','family','aggregation','budget'] if k in frame]
    summaries, details = [], []
    iterable = frame.groupby(groups, dropna=False) if groups else [((),frame)]
    for key, group in iterable:
        key = key if isinstance(key,tuple) else (key,)
        labels = dict(zip(groups,key))
        if group.duplicated(['seed','instance_id','method']).any():
            raise ValueError('Duplicate method/seed/factual records cannot be paired unambiguously')
        methods = sorted(group.method.unique())
        references = [m for m in ['ORPM','ORPM-tree','ORPM_opt','ORPM_eval'] if m in methods]
        for reference in references:
            suffix = '_opt' if reference.endswith('_opt') else '_eval' if reference.endswith('_eval') else None
            for baseline in methods:
                if baseline in references or (suffix and not baseline.endswith(suffix)):
                    continue
                left = group[group.method==reference].set_index(['seed','instance_id'])
                right = group[group.method==baseline].set_index(['seed','instance_id'])
                paired_ids = left.index.intersection(right.index)
                for metric, scale, bounds in [('valid',100.,(-100.,100.)), ('elapsed_seconds',1.,None),
                    ('cost',1.,None), ('max_loss',1.,(-1.,1.)),
                    ('unseen_all_valid',100.,(-100.,100.)), ('unseen_fraction_valid',100.,(-100.,100.)),
                    ('model_fraction_valid',100.,(-100.,100.))]:
                    if metric not in group:
                        continue
                    a = pd.to_numeric(left.loc[paired_ids,metric],errors='coerce')
                    b = pd.to_numeric(right.loc[paired_ids,metric],errors='coerce')
                    diffs = (a.astype(float)-b.astype(float)).replace([np.inf,-np.inf],np.nan).dropna()*scale
                    seed_means = diffs.groupby(level='seed').mean()
                    if seed_means.empty:
                        continue
                    stats = mean_ci(seed_means,bounds)
                    summaries.append({**labels, 'reference':reference,'baseline':baseline,'metric':metric,
                        'direction':'reference minus baseline', 'units':'percentage points' if scale==100 else 'seconds' if metric=='elapsed_seconds' else 'loss' if metric=='max_loss' else 'standardized L2 cost',
                        'paired_instances':len(diffs), 'reference_instances':len(left), 'baseline_instances':len(right),
                        'unpaired_reference_instances':len(left)-len(diffs), 'unpaired_baseline_instances':len(right)-len(diffs),
                        'completed_seeds':stats['n_seeds'],'mean_difference':stats['mean'], **stats})
                    for seed, values in diffs.groupby(level='seed'):
                        details.append({**labels,'reference':reference,'baseline':baseline,'metric':metric,
                            'seed':seed,'paired_instances':len(values),'mean_difference':values.mean()})
    return pd.DataFrame(summaries,columns=None if summaries else base_columns), pd.DataFrame(details)


def save_statistics(records, directory):
    directory = Path(directory)
    seeds = per_seed_table(records)
    paired, paired_seeds = paired_comparisons(records)
    seeds.to_csv(directory/'seed_metrics.csv',index=False)
    paired.to_csv(directory/'paired_comparisons.csv',index=False)
    paired_seeds.to_csv(directory/'paired_seed_differences.csv',index=False)
    return seeds, paired


def plot_paired_validity(run_directory, show=True):
    """Optional forest plot; no uncertainty inferred from overlapping marginal CIs."""
    import matplotlib.pyplot as plt
    path = Path(run_directory)
    data = pd.read_csv(path/'paired_comparisons.csv')
    data = data[data.metric.isin(['valid','unseen_all_valid'])]
    if data.empty:
        print('No paired validity comparisons:',path)
        return None
    fig, ax = plt.subplots(figsize=(10,max(3.,.38*len(data)+1.5)))
    for i, row in enumerate(data.itertuples()):
        ax.plot(row.mean_difference,i,'o',color='#2368a2')
        if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
            ax.plot([row.ci_low,row.ci_high],[i,i],color='#2368a2')
    labels = [f'{r.reference} − {r.baseline}; {r.metric}; b={getattr(r,"budget","n/a")} (S={r.completed_seeds})' for r in data.itertuples()]
    ax.set_yticks(range(len(data)),labels);ax.axvline(0,color='gray',linestyle=':')
    ax.set(xlabel='Paired validity difference (percentage points)',title='Mean and pointwise 95% seed-t CI; unavailable intervals omitted')
    fig.tight_layout();fig.savefig(path/'paired_validity.png',dpi=180,bbox_inches='tight')
    if show:plt.show()
    return fig
