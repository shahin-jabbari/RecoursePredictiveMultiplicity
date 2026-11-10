"""Analytic singleton recourse and audited reuse of the completed linear study.

Usage: python source/run_initial.py [--output results/NEW_NAME]
No training, ORPM optimization, or modification of source results occurs.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import math
import os
import sys
import zipfile
from datetime import datetime, timezone

os.environ.setdefault('MPLCONFIGDIR', '/tmp/multiplicity-matplotlib')
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import t


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def singleton(a, z, free, threshold=0.):
    q = a[free]
    deficit = max(0., threshold - float(z))
    norm = float(np.linalg.norm(q))
    if deficit == 0:
        return 0., np.zeros(len(free))
    if norm == 0:
        return math.inf, None
    return deficit / norm, deficit * q / (norm * norm)


def preflight():
    c, d = singleton(np.array([3., 4., 99.]), -10., np.array([0, 1]))
    assert abs(c - 2.) < 1e-12 and np.allclose(d, [1.2, 1.6])
    assert singleton(np.array([0., 1.]), -1., np.array([0]))[0] == math.inf
    assert singleton(np.array([0., 1.]), 0., np.array([0]))[0] == 0.
    # Orthogonal halfspaces require a common action of sqrt(2), though each
    # separate action costs one. This detects the key conceptual distinction.
    separate = [singleton(np.eye(2)[j], -1., np.arange(2))[0] for j in range(2)]
    assert max(separate) == 1. and np.linalg.norm([1., 1.]) > max(separate)
    # Opposing strict offsets give individually feasible but jointly impossible recourse.
    assert linprog([0.], A_ub=[[-1.], [1.]], b_ub=[-1., -1.],
                   bounds=[(None, None)], method='highs').status == 2


def ci_table(seed_frame, keys, metrics):
    rows = []
    for key, group in seed_frame.groupby(keys, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(keys, key))
        for metric in metrics:
            v = group[metric].dropna().to_numpy(float)
            v = v[np.isfinite(v)]
            n = len(v)
            mean = float(np.mean(v)) if n else np.nan
            half = float(t.ppf(.975, n-1)*np.std(v, ddof=1)/np.sqrt(n)) if n > 1 else np.nan
            rows.append(dict(base, metric=metric, mean=mean, ci_lower=mean-half,
                             ci_upper=mean+half, n_seeds=n, df=n-1 if n else np.nan))
    return pd.DataFrame(rows)


def aggregate(frame, keys, metrics, out, stem):
    seeds = frame.groupby(keys + ['seed'], dropna=False)[metrics].mean().reset_index()
    seeds.to_csv(out / 'tables' / (stem + '_seed_means.csv'), index=False)
    ci = ci_table(seeds, keys, metrics)
    ci.to_csv(out / 'tables' / (stem + '_95ci.csv'), index=False)
    return seeds, ci


def draw(ax, frame, x, metric, label, color):
    z = frame[frame.metric == metric].sort_values(x)
    xx = z[x].to_numpy(float)
    yy = z['mean'].to_numpy(float)
    lo = z.ci_lower.to_numpy(float)
    hi = z.ci_upper.to_numpy(float)
    ax.plot(xx, yy, marker='o', markersize=3, lw=1.8, color=color, label=label)
    ax.fill_between(xx, lo, hi, alpha=.13, color=color)


def plots(success, cost, out, title_suffix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 130})
    for ds in sorted(success.dataset.unique()):
        s = success[(success.dataset == ds) & (success.model_count == 24)]
        fig, ax = plt.subplots(figsize=(7.4, 5.0), layout='constrained')
        for m, label, col in [
            ('single_success', 'Average single model', '#206095'),
            ('separate_success', 'Every model, separate actions', '#2c8b57'),
            ('joint_success_lower', 'One joint action: certified success', '#a44a9e'),
            ('orpm_success', 'ORPM joint success', '#d87418')]:
            draw(ax, s, 'budget', m, label, col)
        ax.set(title=ds.replace('Give_Me_Some_Credit', 'GMC').replace('Folktables', 'ACSIncome') + ' | 24 models',
               xlabel='L2 budget (saved standardized units)', ylabel='Success rate')
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=.18)
        fig.supxlabel('Shading: pointwise 95% t intervals across 10 seed means; eligible-group weights.\nJoint curve is a lower bound if any threshold decision is unresolved.', fontsize=8)
        for ext in ['png', 'pdf']:
            fig.savefig(out / 'figures' / (ds + '_success_24.' + ext))
        plt.close(fig)
        c = cost[cost.dataset == ds]
        fig, ax = plt.subplots(figsize=(7.4, 4.8), layout='constrained')
        for m, label, col in [
            ('premium_mean_upper', 'Joint minus average singleton', '#206095'),
            ('premium_max_upper', 'Joint minus hardest singleton', '#a44a9e')]:
            draw(ax, c, 'model_count', m, label, col)
        ax.set(title=ds.replace('Give_Me_Some_Credit', 'GMC').replace('Folktables', 'ACSIncome') + ' | capped cost premium',
               xlabel='Number of competing models', ylabel='Additional capped L2 distance', xticks=[6, 12, 24])
        ax.legend(fontsize=9)
        ax.grid(alpha=.18)
        fig.supxlabel('Pointwise 95% t intervals; tau = largest saved budget.\nNumerical upper-bound estimates shown; lower bounds are included in tables.', fontsize=8)
        for ext in ['png', 'pdf']:
            fig.savefig(out / 'figures' / (ds + '_cost_premium.' + ext))
        plt.close(fig)


def report(out, title, paragraphs, tables):
    text = title + '\n\n' + '\n\n'.join(paragraphs)
    sections = []
    for label, table in tables:
        sections.append('<h2>' + html.escape(label) + '</h2>' + table.to_html(index=False, float_format=lambda v: f'{v:.5g}'))
        text += '\n\n' + label + '\n' + table.to_string(index=False, float_format=lambda v: f'{v:.5g}')
    (out / 'report.txt').write_text(text + '\n')
    figures = ''.join('<figure><img src="figures/' + p.name + '" alt="' + p.stem + '"><figcaption>' + p.stem + '</figcaption></figure>'
                      for p in sorted((out / 'figures').glob('*.png')))
    (out / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>' + html.escape(title) + '</title>'
        '<style>body{font:16px/1.55 system-ui;max-width:1120px;margin:40px auto;padding:0 24px;color:#17283b}h1{font-size:30px}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:5px 9px;border-bottom:1px solid #ddd;text-align:right}img{width:100%;max-width:850px}figure{margin:30px 0}figcaption{font-size:12px;color:#536170}</style><h1>'
        + html.escape(title) + '</h1>' + ''.join('<p>' + html.escape(p) + '</p>' for p in paragraphs) + ''.join(sections) + figures + '</html>')
    with zipfile.ZipFile(out / 'figures_tables.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for folder in ['figures', 'tables']:
            for p in sorted((out / folder).iterdir()):
                z.write(p, p.relative_to(out))
        for name in ['report.txt', 'report.html', 'protocol.json', 'verification.json']:
            z.write(out / name, name)


def main():
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cfg = json.loads((here / 'analysis_protocol.json').read_text())
    src = Path(cfg['source_result'])
    out = args.output or here / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    for name in ['tables', 'figures']:
        (out / name).mkdir()
    save(out / 'protocol.json', cfg)
    hashes = {}
    def track(path):
        path = Path(path)
        if str(path) not in hashes:
            hashes[str(path)] = sha(path)
        return path
    def read(path):
        return json.loads(track(path).read_text())
    def check(ok, label):
        if not bool(ok):
            raise AssertionError(label)
    preflight()
    check(read(src / 'preparation_verification.json')['passed'], 'Original verification must pass')
    source_cfg = read(src / 'protocol.json')
    check(source_cfg['datasets'] == cfg['datasets'] and source_cfg['budgets'] == cfg['budgets'], 'Source design drift')
    for name in ['analysis_protocol.json', 'PROTOCOL.txt', 'source/analyze_cost.py']:
        track(here / name)
    save(out / 'freeze.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
                              'protocol_sha256': sha(here / 'analysis_protocol.json'), 'code_sha256': sha(Path(__file__))})
    # Only ORPM records are needed; exact benchmark certificates are read directly.
    orpm = {}
    with open(track(src / 'records.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            if r['method'] == 'ORPM':
                key = tuple(r[k] for k in ['dataset', 'seed', 'person', 'model_count', 'budget_index'])
                check(key not in orpm, 'Duplicate ORPM key')
                orpm[key] = r
    check(len(orpm) == cfg['expected_budget_settings'], 'ORPM coverage')
    singles, costs, settings, groups, inventory = [], [], [], [], []
    tol = cfg['logit_tolerance']
    maximum_cost_bound_width = 0.
    joint_threshold_unresolved = 0
    for ds in cfg['datasets']:
        for seed in cfg['seeds']:
            block = src / 'blocks' / ds / f'seed_{seed}'
            pre = read(block / 'bank/preprocessing.json')
            cohort = read(block / 'cohort.json')
            with np.load(track(block / 'panel.npz')) as p:
                A, b, X, free, ids = [p[k] for k in ['A', 'b', 'selected_X', 'free', 'test_ids']]
            with np.load(track(block / 'bank/split_and_predictions.npz')) as p:
                Xtest = p['X_test'].astype(np.float32).astype(float)
                allids = p['test_ids']
            a2, b2 = [], []
            for j in range(24):
                with np.load(track(block / 'bank' / f'model_{j:04d}.npz')) as p:
                    prefix = 'base_model.' if 'feature_mask' in p else ''
                    aa = p[prefix + 'linear.weight'].ravel()
                    aa = aa * (p['feature_mask'] if 'feature_mask' in p else np.ones_like(aa))
                    a2.append(aa.astype(float))
                    b2.append(float(p[prefix + 'linear.bias'].item()))
            check(np.array_equal(A, a2) and np.array_equal(b, b2), f'Coefficient reconstruction {ds}/{seed}')
            check(np.array_equal(free, pre['actionable_indices']), 'Action region')
            test_logits = Xtest @ A.T + b
            eligible = np.flatnonzero(np.logical_and.reduce([(test_logits[:, :n] < 0).sum(1) >= n/2 for n in cfg['model_counts']]))
            order = np.random.default_rng(seed + source_cfg['cohort_order_seed_offset']).permutation(eligible)
            check(cohort['random_order'] == order.tolist(), 'Saved sampling order')
            selected = np.asarray(cohort['selected_indices'], dtype=int)
            check(cohort['selected_indices'][:20] == order[:20].tolist(), 'Preserved cohort prefix')
            check(np.array_equal(X, Xtest[selected]) and np.array_equal(ids, allids[selected]), 'Expanded cohort identities')
            immutable = np.setdiff1d(np.arange(A.shape[1]), free)
            group_cfg = cfg['groups'].get(ds)
            group_values = None
            if group_cfg:
                ix = pre['feature_names'].index(group_cfg['feature'])
                check(ix in immutable, 'Group attribute must be immutable')
                raw = Xtest[:, ix] * pre['scale'][ix] + pre['mean'][ix]
                check(np.max(abs(raw - np.round(raw))) < 1e-4, 'Group values must reconstruct integer codes/ages')
                raw = np.round(raw).astype(int)
                if group_cfg['kind'] == 'coded':
                    check(set(np.unique(raw)) <= set(group_cfg['codes']), 'Unrecognized group code')
                    first = raw == group_cfg['first_code']
                else:
                    first = raw < group_cfg['threshold']
                allgroups = np.where(first, group_cfg['first'], group_cfg['second'])
                group_values = allgroups[selected]
                for name in [group_cfg['first'], group_cfg['second']]:
                    inventory.append(dict(dataset=ds, seed=seed, group=name,
                                          heldout_count=int(np.sum(allgroups == name)),
                                          eligible_count=int(np.sum(allgroups[eligible] == name)),
                                          selected_count=int(np.sum(group_values == name))))
            for person, x0 in enumerate(X):
                z = A @ x0 + b
                c = np.empty(24)
                ct = np.empty(24)
                points = np.tile(x0, (24, 1))
                for j in range(24):
                    c[j], delta = singleton(A[j], z[j], free)
                    ct[j], _ = singleton(A[j], z[j], free, -tol)
                    if delta is not None:
                        points[j, free] += delta
                        check(abs(np.linalg.norm(points[j] - x0) - c[j]) <= 1e-8*(1+c[j]), 'Singleton cost')
                        check(A[j] @ points[j] + b[j] >= -tol, 'Singleton acceptance')
                        check(np.array_equal(points[j, immutable], x0[immutable]), 'Singleton immutable coordinates')
                    singles.append(dict(dataset=ds, seed=seed, person=person, test_id=int(ids[person]),
                                        model=j, nominal_cost=c[j], evaluation_cost=ct[j],
                                        initially_accepted=bool(z[j] >= -tol), feasible=bool(np.isfinite(c[j]))))
                groups.append(dict(dataset=ds, seed=seed, person=person, test_id=int(ids[person]),
                                   group=group_values[person] if group_values is not None else 'not_defined'))
                transfer = points @ A.T + b >= -tol
                for n in cfg['model_counts']:
                    case = block / 'cases' / f'person_{person:02d}_n_{n}'
                    q = read(case / 'minimum_cost.json')
                    info = q['info']
                    Q = A[:n, free]
                    if info['available']:
                        point = np.asarray(q['point'])
                        lam = np.asarray(info['dual_multipliers'])
                        delta = point[free] - x0[free]
                        primal = .5 * delta @ delta
                        dual = -lam @ z[:n] - .5 * np.sum((Q.T @ lam)**2)
                        check(np.all(lam >= 0) and np.min(A[:n] @ point + b[:n]) >= -tol, 'Joint cost primal/dual feasibility')
                        check(np.array_equal(point[immutable], x0[immutable]), 'Joint cost immutability')
                        check(-1e-7 <= primal-dual , 'Joint cost certificate')
                        check(abs(primal-info['primal_objective']) < 1e-7 and abs(dual-info['dual_objective']) < 1e-7, 'Saved cost certificate values')
                        low, high = np.sqrt(max(0., 2*dual)), np.sqrt(max(0., 2*primal))
                        maximum_cost_bound_width = max(maximum_cost_bound_width, high-low)
                    else:
                        lp = linprog(np.zeros(len(free)), A_ub=-Q, b_ub=z[:n],
                                     bounds=[(None, None)]*len(free), method='highs')
                        if info['status'] == 'certified_infeasible':
                            check(lp.status == 2 and info['certificate_passed'], 'Independent infeasibility')
                            low = high = math.inf
                        else:
                            check(info['status'] == 'unresolved', 'Explicit unresolved outcome')
                            low, high = 0., math.inf
                    tau = max(cfg['budgets'][ds])
                    cm = np.minimum(c[:n], tau)
                    cl, cu = min(low, tau), min(high, tau)
                    finite = np.isfinite(c[:n])
                    allfinite = bool(finite.all() and np.isfinite(high))
                    check(not allfinite or high + 1e-6 >= max(c[:n]), 'Joint cost cannot beat hardest singleton')
                    sub = transfer[:n, :n]
                    rejected = z[:n] < -tol
                    valid_origins = finite
                    valid_rejected = finite & rejected
                    cross = (sub.sum(1) - np.diag(sub)) / (n-1)
                    costs.append(dict(dataset=ds, seed=seed, person=person, model_count=n, tau=tau,
                        joint_cost_lower=low, joint_cost_upper=high, single_mean_cost=float(c[:n].mean()),
                        single_max_cost=float(c[:n].max()), joint_capped_lower=cl, joint_capped_upper=cu,
                        single_capped_mean=float(cm.mean()), single_capped_max=float(cm.max()),
                        premium_mean_lower=cl-cm.mean(), premium_mean_upper=cu-cm.mean(),
                        premium_max_lower=cl-cm.max(), premium_max_upper=cu-cm.max(),
                        normalized_premium_mean_lower=(cl-cm.mean())/tau,
                        normalized_premium_mean_upper=(cu-cm.mean())/tau,
                        uncapped_premium_mean=high-c[:n].mean() if allfinite else np.nan,
                        uncapped_premium_max=high-c[:n].max() if allfinite else np.nan,
                        finite_comparison=float(allfinite), joint_infeasible=float(info['status'] == 'certified_infeasible'),
                        joint_above_horizon_lower=float(low > tau), joint_above_horizon_upper=float(high > tau),
                        singleton_infeasible_fraction=float(np.mean(~finite)),
                        transfer_mean_other=float(cross[valid_origins].mean()) if valid_origins.any() else np.nan,
                        transfer_joint=float(sub.all(1)[valid_origins].mean()) if valid_origins.any() else np.nan,
                        transfer_mean_other_rejected=float(cross[valid_rejected].mean()) if valid_rejected.any() else np.nan,
                        transfer_joint_rejected=float(sub.all(1)[valid_rejected].mean()) if valid_rejected.any() else np.nan,
                        available_singleton_fraction=float(finite.mean()),
                        rejected_model_count=int(rejected.sum())))
                    for bi, budget in enumerate(cfg['budgets'][ds]):
                        r = orpm[(ds, seed, person, n, bi)]
                        check(float(r['budget']) == float(budget) and r['test_id'] == int(ids[person]), 'Paired ORPM identity')
                        with np.load(track(src / r['point_file'])) as f:
                            op = f['ORPM']
                        ocost = np.linalg.norm(op-x0)
                        oz = A[:n] @ op + b[:n]
                        check(ocost <= budget+1e-8 and np.array_equal(op[immutable], x0[immutable]), 'ORPM action feasibility')
                        oval = float(bool(r['available']) and np.all(oz >= -tol))
                        check(oval == r['seen_joint_validity'] and abs(ocost-r['cost']) < 1e-9, 'ORPM evaluated outcome')
                        margin = read(case / f'budget_{bi}' / 'exact_margin.json')
                        point, mi = np.asarray(margin['point']), margin['info']
                        w = np.asarray(mi['dual_weights'])
                        lower = float(min(A[:n] @ point + b[:n]))
                        upper = float(w @ z[:n] + budget * np.linalg.norm(Q.T @ w)) if budget else float(min(z[:n]))
                        check(np.all(w >= 0) and abs(sum(w)-1) < 1e-9, 'Margin dual weights')
                        check(np.linalg.norm(point-x0) <= budget+1e-8 and np.array_equal(point[immutable], x0[immutable]), 'Margin primal feasibility')
                        check(-1e-7 <= upper-lower, 'Valid margin bound ordering')
                        check(abs(lower-mi['margin_lower']) < 1e-8 and abs(upper-mi['margin_upper']) < 1e-8, 'Saved margin bounds')
                        jl = float(lower >= -tol or oval == 1)
                        ju = float(upper >= -tol or jl == 1)
                        joint_threshold_unresolved += int(jl != ju)
                        single_ok = ct[:n] <= budget+1e-8
                        ss, separate = float(single_ok.mean()), float(single_ok.all())
                        check(jl <= separate and separate <= ss+1e-12, 'Success hierarchy')
                        settings.append(dict(dataset=ds, seed=seed, person=person, model_count=n,
                            budget=budget, budget_index=bi, single_success=ss, separate_success=separate,
                            joint_success_lower=jl, joint_success_upper=ju, orpm_success=oval,
                            initial_single_acceptance=float(np.mean(~rejected)),
                            single_success_rejected=float(single_ok[rejected].mean()) if rejected.any() else np.nan,
                            harder_models_penalty=ss-separate,
                            common_action_penalty_lower=separate-ju, common_action_penalty_upper=separate-jl,
                            structural_penalty_lower=ss-ju, structural_penalty_upper=ss-jl,
                            algorithmic_shortfall_lower=jl-oval, algorithmic_shortfall_upper=ju-oval,
                            operational_penalty=ss-oval, joint_unresolved=ju-jl,
                            orpm_returned_cost=float(ocost)))
            print(f'Completed {ds} seed {seed}', flush=True)
    check(len(singles) == cfg['expected_singleton_solutions'] and len(costs) == cfg['expected_cost_comparisons'] and len(settings) == cfg['expected_budget_settings'], 'Complete analysis counts')
    sf, cf, bf = pd.DataFrame(singles), pd.DataFrame(costs), pd.DataFrame(settings)
    gf = pd.DataFrame(groups)
    sf.to_csv(out / 'singleton_costs.csv', index=False)
    cf.to_csv(out / 'cost_comparisons.csv', index=False)
    bf.to_csv(out / 'budget_comparisons.csv', index=False)
    gf.to_csv(out / 'group_labels.csv', index=False)
    pd.DataFrame(inventory).to_csv(out / 'tables/group_count_audit.csv', index=False)
    bm = [c for c in bf if c not in ['dataset', 'seed', 'person', 'model_count', 'budget', 'budget_index', 'orpm_returned_cost']]
    cm = [c for c in cf if c not in ['dataset', 'seed', 'person', 'model_count', 'tau', 'joint_cost_lower', 'joint_cost_upper', 'single_mean_cost', 'single_max_cost', 'rejected_model_count']]
    from expansion_analysis import weighted_aggregate, preserve_comparison, paired_effects, more_plots
    preserve_comparison(out, cfg)
    _, bc = weighted_aggregate(bf, gf, pd.DataFrame(inventory), ['dataset', 'model_count', 'budget'], bm, out, 'success')
    _, cc = weighted_aggregate(cf, gf, pd.DataFrame(inventory), ['dataset', 'model_count'], cm, out, 'cost')
    paired_effects(out, 'success', ['dataset', 'model_count', 'budget'], bm)
    paired_effects(out, 'cost', ['dataset', 'model_count'], cm)
    plots(bc, cc, out)
    more_plots(bc, cc, out)
    for path, digest in hashes.items():
        check(sha(path) == digest, 'Input changed during analysis: ' + path)
    save(out / 'input_sha256.json', hashes)
    verification = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        singleton_solutions=len(sf), person_model_comparisons=len(cf), budget_settings=len(bf),
        source_inputs_unchanged=True, independent_coefficients_and_cohorts=True,
        recomputed_cost_and_margin_certificates=True, orpm_points_rechecked=True,
        joint_threshold_unresolved=joint_threshold_unresolved,
        maximum_finite_cost_certificate_width=maximum_cost_bound_width,
        numerical_scope='Float64 checks at frozen tolerances, not interval arithmetic',
        source_orpm_traces='Preserved original traces; new traces independently checked in worker verification', stratification_weights='Within-seed eligible group fractions; seed means equally weighted')
    save(out / 'verification.json', verification)
    largest = bf.groupby('dataset').budget.max().to_dict()
    headline = bc[(bc.model_count == 24) & bc.apply(lambda r: r.budget == largest[r.dataset], axis=1)
                  & bc.metric.isin(['single_success', 'separate_success', 'joint_success_lower', 'joint_success_upper', 'orpm_success', 'structural_penalty_upper', 'algorithmic_shortfall_upper'])]
    pars = [
        f"Linear expansion completed: {len(sf):,} singleton costs, {len(cf):,} joint-cost comparisons, {len(bf):,} budget settings. Original 1,200 cases are included without re-solving; 633 new person-seed cases use the same frozen linear models and action regions.",
        "Expanded COMPAS and ACSIncome cohorts target 20 people per recorded-sex group and seed. German and GMC target 20 per age group (under 40 versus 40 and older), using all eligible people when a group is smaller. Original twenty-person cohorts remain a prefix. Synthetic and Polish retain their original cohorts.",
        "Primary overall estimates weight the two group means by each seed's eligible group proportions before equally averaging seeds. Expanded samples are stratified; unweighted pooled summaries are not population estimates. The original cohort summaries are reported separately. Phase contrasts are paired and overlap; they are not independent replications.",
        "Average singleton success permits a model-specific recommendation; every-model-separately success permits distinct recommendations; joint optimum and ORPM require the same action to satisfy all competing models. These comparisons separate harder models, common-action burden, and ORPM shortfall.",
        "Costs are nominal minimum standardized L2 distances. Capped cost min(cost,tau), tau=largest dataset budget, includes every person; the cap is not an imputed true cost. ORPM optimizes worst-model loss within a budget, not minimum distance. Uncertified or unavailable outcomes retain numerical bounds; no incumbent is not evidence of infeasibility.",
        "All inferential figures and tables have descriptive pointwise 95% Student t intervals across paired seed means (df=9 for ten seeds). Numerical bounds are separate from statistical intervals. Unresolved joint budget decisions: " + str(joint_threshold_unresolved) + ". Maximum finite numerical cost-bound width: " + f"{maximum_cost_bound_width:.5g}" + ".",
        "Models, budgets and people repeated across saved splits are not independent samples. Results are conditional on these benchmark datasets, eligible majority-rejection cohorts, model masks, splits and continuous immutable-feature action regions. Model count alone does not isolate a causal effect of predictive multiplicity, and L2 distance is not monetary or causal burden. No new training or unseen-model experiment was conducted.",
    ]
    report(out, 'Cost of multiplicity: expanded linear cohorts', pars,
           [('Largest budget, 24 models (weighted estimates and pointwise 95% intervals)', headline),
            ('Capped cost premiums (weighted estimates and pointwise 95% intervals)', cc[cc.metric.isin(['premium_mean_upper', 'premium_max_upper'])])])

    save(here / 'latest.json', {'result': str(out.resolve()), 'status': 'completed', 'verification_passed': True})
    print('RESULT ' + str(out.resolve()), flush=True)


if __name__ == '__main__':
    main()
