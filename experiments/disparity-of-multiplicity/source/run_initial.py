"""Exploratory group comparisons using the verified sibling cost study.
Usage: python source/run_initial.py [--cost-result PATH] [--output NEW_PATH]
"""
from pathlib import Path
import argparse
import importlib.util
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd


def main():
    here = Path(__file__).resolve().parents[1]
    cost_study = here.parent / 'cost-of-multiplicity'
    spec = importlib.util.spec_from_file_location('cost_analysis', cost_study / 'source/run_initial.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cost-result', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.cost_result or Path(json.loads((cost_study / 'latest.json').read_text())['result'])
    out = args.output or here / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    for name in ['tables', 'figures']:
        (out / name).mkdir()
    cfg = json.loads((here / 'protocol.json').read_text())
    common.save(out / 'protocol.json', cfg)
    assert json.loads((source / 'verification.json').read_text())['passed']
    inputs = [source / n for n in ['verification.json', 'protocol.json', 'budget_comparisons.csv',
              'cost_comparisons.csv', 'group_labels.csv', 'tables/group_count_audit.csv']]
    inputs += [here / 'protocol.json', here / 'PROTOCOL.txt', Path(__file__), cost_study / 'source/run_initial.py']
    hashes = {str(p): common.sha(p) for p in inputs}
    common.save(out / 'freeze.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
                                    'input_sha256': hashes})
    labels = pd.read_csv(source / 'group_labels.csv')
    counts = pd.read_csv(source / 'tables/group_count_audit.csv')
    counts.to_csv(out / 'tables/group_count_audit.csv', index=False)
    labels.to_csv(out / 'group_labels.csv', index=False)
    count_table = counts.groupby(['dataset', 'group'])[['selected_count', 'eligible_count', 'heldout_count']].sum().reset_index()
    paired_counts = []
    for ds, g in cfg['groups'].items():
        z = counts[counts.dataset == ds].pivot(index='seed', columns='group', values='selected_count')
        complete = (z[g['first']] > 0) & (z[g['second']] > 0)
        for seed, row in z.iterrows():
            paired_counts.append(dict(dataset=ds, seed=int(seed), first_group=g['first'], second_group=g['second'],
                                      first_count=int(row[g['first']]), second_count=int(row[g['second']]),
                                      included=bool(complete.loc[seed])))
    pair_counts = pd.DataFrame(paired_counts)
    pair_counts.to_csv(out / 'tables/paired_seed_audit.csv', index=False)
    cis = {}
    for stem, filename in [('success', 'budget_comparisons.csv'), ('cost', 'cost_comparisons.csv')]:
        data = pd.read_csv(source / filename).merge(labels[['dataset', 'seed', 'person', 'group']],
                                                   on=['dataset', 'seed', 'person'], validate='many_to_one')
        data = data[data.dataset.isin(cfg['groups'])]
        keys = ['dataset', 'model_count'] + (['budget'] if stem == 'success' else [])
        ignore = ['dataset', 'seed', 'person', 'model_count', 'budget', 'budget_index', 'group', 'tau',
                  'orpm_returned_cost', 'joint_cost_lower', 'joint_cost_upper', 'single_mean_cost',
                  'single_max_cost', 'rejected_model_count']
        metrics = [c for c in data if c not in ignore]
        seeds, diffs = [], []
        for ds, g in cfg['groups'].items():
            complete = pair_counts[(pair_counts.dataset == ds) & pair_counts.included].seed.tolist()
            z = data[(data.dataset == ds) & data.seed.isin(complete)]
            means = z.groupby(keys + ['seed', 'group'])[metrics].mean().reset_index()
            first = means[means.group == g['first']].set_index(keys + ['seed'])
            second = means[means.group == g['second']].set_index(keys + ['seed'])
            assert first.index.equals(second.index)
            # Secondary finite-only quantities can be undefined even when both
            # groups exist. Pair their available seeds too, not just group counts.
            for metric in metrics:
                ok = np.isfinite(first[metric]) & np.isfinite(second[metric])
                first.loc[~ok, metric] = np.nan
                second.loc[~ok, metric] = np.nan
            seeds.append(pd.concat([first.reset_index(), second.reset_index()], ignore_index=True))
            diff = first[metrics].copy()
            # Bound subtraction reverses the second group's bound endpoints.
            for metric in metrics:
                opposite = metric
                if metric.endswith('_lower') and metric[:-6] + '_upper' in metrics:
                    opposite = metric[:-6] + '_upper'
                elif metric.endswith('_upper') and metric[:-6] + '_lower' in metrics:
                    opposite = metric[:-6] + '_lower'
                diff[metric] = first[metric] - second[opposite]
            diff = diff.reset_index()
            diff['contrast'] = g['first'] + '_minus_' + g['second']
            diffs.append(diff)
        sm, dm = pd.concat(seeds, ignore_index=True), pd.concat(diffs, ignore_index=True)
        sm.to_csv(out / 'tables' / (stem + '_group_seed_means.csv'), index=False)
        dm.to_csv(out / 'tables' / (stem + '_gap_seed_means.csv'), index=False)
        gc = common.ci_table(sm, keys + ['group'], metrics)
        dc = common.ci_table(dm, keys + ['contrast'], metrics)
        gc.to_csv(out / 'tables' / (stem + '_group_95ci.csv'), index=False)
        dc.to_csv(out / 'tables' / (stem + '_gap_95ci.csv'), index=False)
        cis[stem] = dc
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    for ds, g in cfg['groups'].items():
        nseeds = int(pair_counts[(pair_counts.dataset == ds) & pair_counts.included].shape[0])
        s = cis['success'][(cis['success'].dataset == ds) & (cis['success'].model_count == 24)]
        fig, ax = plt.subplots(figsize=(7.8, 5), layout='constrained')
        for metric, label, color in [('structural_penalty_upper', 'Difference in structural penalties', '#206095'),
                                     ('operational_penalty', 'Difference in singleton-to-ORPM penalties', '#d87418')]:
            common.draw(ax, s, 'budget', metric, label, color)
        ax.axhline(0, color='#555', lw=.7)
        ax.set(title=ds.replace('Folktables', 'ACSIncome').replace('Give_Me_Some_Credit', 'GMC') + ' | 24 models',
               xlabel='L2 budget', ylabel='First-group minus second-group penalty')
        ax.legend(fontsize=8)
        ax.grid(alpha=.18)
        fig.supxlabel(f"{g['first']} minus {g['second']}; {nseeds} complete-pair seeds.\nPointwise descriptive 95% t intervals; existing selected cohorts only.", fontsize=8)
        for ext in ['png', 'pdf']:
            fig.savefig(out / 'figures' / (ds + '_success_penalty_gap.' + ext), dpi=150)
        plt.close(fig)
        c = cis['cost'][cis['cost'].dataset == ds]
        fig, ax = plt.subplots(figsize=(7.8, 5), layout='constrained')
        for metric, label, color in [('single_capped_mean', 'Singleton capped-cost gap', '#2c8b57'),
                                     ('joint_capped_upper', 'Joint capped-cost gap', '#a44a9e'),
                                     ('premium_mean_upper', 'Additional multiplicity cost gap', '#206095')]:
            common.draw(ax, c, 'model_count', metric, label, color)
        ax.axhline(0, color='#555', lw=.7)
        ax.set(title=ds.replace('Folktables', 'ACSIncome').replace('Give_Me_Some_Credit', 'GMC') + ' | capped cost disparity',
               xlabel='Number of competing models', ylabel='First-group minus second-group cost', xticks=[6, 12, 24])
        ax.legend(fontsize=8)
        ax.grid(alpha=.18)
        fig.supxlabel(f"{g['first']} minus {g['second']}; {nseeds} complete-pair seeds.\nPointwise 95% t intervals; numerical upper-bound differences shown.", fontsize=8)
        for ext in ['png', 'pdf']:
            fig.savefig(out / 'figures' / (ds + '_cost_gap.' + ext), dpi=150)
        plt.close(fig)
    for path, digest in hashes.items():
        assert common.sha(path) == digest, 'Input drift: ' + path
    common.save(out / 'verification.json', dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        source_cost_verification_passed=True, input_preservation=True, group_attributes_immutable=True,
        coding_audit='Performed in linked cost study; saved reconstructed labels reused',
        complete_pair_seeds=pair_counts.groupby('dataset').included.sum().astype(int).to_dict(),
        no_new_training_or_recourse=True, status='exploratory_existing_cohort'))
    common.save(out / 'input_sha256.json', hashes)
    pars = [
        'This exploratory analysis separates baseline singleton disparity, the additional structural burden of one common action, and ORPM shortfall. The four group audits use exactly the same saved people and models as the linked cost study.',
        'Recorded sex is used for COMPAS and ACSIncome. German and GMC use a prespecified descriptive age split: under 40 versus 40 and older. German sex is not inferred from its combined personal-status field. No protected-group analysis is claimed for Polish or Synthetic.',
        'Positive penalty differences mean the first group incurs a larger multiplicity penalty; positive success differences mean greater success for the first group. Cost differences are in standardized L2 units and are not monetary or causal burdens.',
        'Group means and differences use the same complete-pair seeds. A seed with either group absent is omitted from both group means and the comparison. Every inferential table reports pointwise descriptive 95% t intervals, n_seeds and degrees of freedom. Raw count tables are exact audit counts.',
        'The original COMPAS cohort has only 15 female-coded person–seed observations; one seed has none. These are not necessarily distinct people. That comparison is too sparse for a strong population fairness conclusion. The protocol specifies an additional stratified cohort phase; no extra recourse or model training was launched here.',
        'The source majority-rejection cohorts, class balancing and repeated splits limit the target population. Group comparisons are conditional on the saved benchmark data, not population-representative or causal. Age cutoffs were fixed for this analysis, not optimized for the observed differences.',
        'Numerical lower and upper bounds are separate from statistical intervals. The plots show upper endpoints for bounded numerical quantities, with both endpoints retained in tables. The previous observations are reused, not independent replication.',
    ]
    summary = cis['cost'][(cis['cost'].model_count == 24) & cis['cost'].metric.isin(['single_capped_mean', 'joint_capped_upper', 'premium_mean_upper'])]
    common.report(out, 'Disparity of multiplicity: initial linear cohort analysis', pars,
                  [('Group counts across seeds (person–seed records, not unique people)', count_table),
                   ('Capped cost differences, 24 models (pointwise 95% intervals)', summary)])
    common.save(here / 'latest.json', {'result': str(out.resolve()), 'status': 'completed_exploratory', 'verification_passed': True})
    print('RESULT ' + str(out.resolve()), flush=True)


if __name__ == '__main__':
    main()
