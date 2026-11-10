"""Post-hoc neural/mixed gap evaluation with revocable numerical bound sources."""
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from utils import recourse_results as rr, recourse_gmc_mixed as mixed
from utils import recourse_convergence as gap
from utils.recourse_duality import check_feasible
from utils.recourse_bound_checks import DualBoundLedger, DIAGNOSTIC_VERSION


def read(path):return json.loads(Path(path).read_text())
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def archive_legacy(runner,folder):
    """Move old diagnostics outside blocks so reports cannot double-count them."""
    checkpoints=folder/'checkpoints'
    files=list(checkpoints.glob('*.json')) if checkpoints.exists() else []
    if not files or all(read(p).get('diagnostic_version')==DIAGNOSTIC_VERSION for p in files):return
    target=runner.root/'diagnostic_revisions'/'before_witness_checked_v2'/folder.relative_to(runner.root)
    target.mkdir(parents=True,exist_ok=True)
    for name in ['checkpoints','gaps.csv','completed.json','bound_audit.json']:
        old=folder/name
        if old.exists():
            if (target/name).exists():raise RuntimeError(f'Diagnostic backup already exists: {target/name}')
            os.replace(old,target/name)


def diagnostics(runner,folder,models,origin,budget,constraints,common_B,z,infos,identity):
    archive_legacy(runner,folder)
    local=runner.local(models,origin,budget,constraints,True)
    points=z['points'];losses=z['losses'];weights=z['weights']
    check_feasible(points,origin,budget,constraints)
    component=np.array([float(np.logaddexp(0,-mixed.layer_bounds(m,origin,budget,constraints)[-1][1][0])) for m in models])
    set_B=mixed.common_normalizer([models],origin,budget,constraints)
    # This is retrospective gap evaluation, not an online stopping rule. Later
    # responses may expose an invalid earlier bound; they never change ORPM.
    witness_points=list(points.copy())+[origin.copy()]
    ledger=DualBoundLedger(np.vstack([losses,local.losses(origin)]))
    records=[];elapsed=0.;evaluator=None
    directory=folder/'checkpoints';directory.mkdir(exist_ok=True)

    def witness(point):
        check_feasible(point,origin,budget,constraints)
        witness_points.append(np.asarray(point,float).copy())
        ledger.add_witnesses(local.losses(point))

    def add(record,aux):
        witness(aux['average_point']);witness(aux['evaluation_response']);witness(aux['dual_response'])
        wb=aux['average_weights'];t=record['round']
        ledger.add_source(t,wb,max(0.,float(wb@component)-1e-12),label=f'component_{t}')
        globalinfo=record['global_evaluation']
        if globalinfo.get('bound_consistent') and np.isfinite(globalinfo['lower_bound']):
            ledger.add_source(t,wb,globalinfo['lower_bound'],numerical=True,label=f'solver_{t}')
        records.append((record,aux))

    def persist():
        allrows=[]
        for record,aux in records:
            t=record['round'];wb=aux['average_weights']
            lower,upper,source=ledger.interval(t,wb)
            best_index=int(np.argmin(ledger.losses@wb))
            aux['dual_response']=witness_points[best_index].copy()
            played=record['played_loss'];average=aux['average_response_losses'];point_loss=aux['point_losses']
            current=[]
            invalidated=any(s['invalidated'] and s['round']<=t for s in ledger.sources)
            for kind,primal in [('averaged_point',float(point_loss.max())),('randomized_response',float(average.max()))]:
                interval=gap.gap_interval(primal,lower,upper,tolerance=1e-6)
                row=dict(identity,round=t,gap_kind=kind,primal=primal,dual_lower=lower,dual_upper=upper,
                    gap_lower=interval['gap_lower_bound'],gap_upper=interval['gap_upper_bound'],dual_uncertainty=upper-lower,
                    normalized_gap_lower=interval['gap_lower_bound']/common_B,normalized_gap_upper=interval['gap_upper_bound']/common_B,
                    common_normalizer=common_B,set_loss_bound=set_B,raw_outer_rate=runner.c['raw_outer_rate'],
                    adversary_regret=float(average.max())-played,recourse_regret_lower=played-upper,recourse_regret_upper=played-lower,
                    averaging_difference=float(point_loss.max()-average.max()),algorithm_seconds=float(z['algorithm_seconds'][t-1]),
                    diagnostic_seconds=record['diagnostic_elapsed_seconds'],global_solver_seconds=record['global_solver_seconds'],
                    global_build_seconds=record['global_build_seconds'],convex_theory_upper=None,mean_oracle_error_bound=None,
                    mean_inner_iterations=float(np.mean([i['optimizer_iterations'] for i in infos[:t]])),
                    oracle_success_fraction=float(np.mean([i['optimizer_success'] for i in infos[:t]])),
                    oracle_hit_cap_fraction=float(np.mean([i['optimizer_iterations']>=runner.c['oracle_maxiter'] for i in infos[:t]])),
                    global_bound_discarded=invalidated or record['global_evaluation'].get('bound_consistent') is False,
                    numerical_tolerance=1e-6,diagnostic_version=DIAGNOSTIC_VERSION,dual_lower_source=source,
                    evaluation_scope='post_hoc_all_saved_feasible_responses',artifact=str(folder.relative_to(runner.root)))
                current.append(row)
            record.update(rows=current,dual_lower=lower,dual_upper=upper,lower_source=source,
                          numerical_source_invalidated=invalidated,diagnostic_version=DIAGNOSTIC_VERSION)
            rr._npz(directory/f'{t:04d}.npz',aux)
            rr.atomic_json(directory/f'{t:04d}.json',record)
            allrows.extend(current)
        rr.atomic_json(folder/'bound_audit.json',dict(diagnostic_version=DIAGNOSTIC_VERSION,
            sources=ledger.sources,invalidations=ledger.invalidations,witness_count=len(ledger.losses),
            qualification='Witness checks can disprove a numerical bound; absence of a contradiction is not a mathematical proof.'))
        return allrows

    try:
        # Rebuild provenance on resume, including any later saved witness, before
        # using an earlier bound. Rewriting rows also repairs interrupted saves.
        for t in runner.c['checkpoints']:
            saved=directory/f'{t:04d}.json'
            if not saved.exists():continue
            record=read(saved);aux=arrays(saved.with_suffix('.npz'))
            if record.get('diagnostic_version')!=DIAGNOSTIC_VERSION:raise RuntimeError('Unexpected diagnostic version')
            add(record,aux);elapsed=max(elapsed,record['diagnostic_elapsed_seconds'])
        if records:persist()
        done={r['round'] for r,a in records}
        for t in runner.c['checkpoints']:
            if t in done:continue
            runner.progress('duality_gap',current=str(folder.relative_to(runner.root)),round=t,diagnostic_version=DIAGNOSTIC_VERSION)
            started=time.perf_counter();xb=points[:t].mean(axis=0);wb=weights[:t].mean(axis=0)
            average=losses[:t].mean(axis=0);point_loss=local.losses(xb);witness(xb)
            initial=witness_points[int(np.argmin(ledger.losses@wb))]
            response,localinfo=local.solve(wb,initial);witness(response)
            analytic=max(0.,float(wb@component)-1e-12)
            inherited,upper,_=ledger.interval(t,wb);lower=max(analytic,inherited)
            globalinfo={};build_seconds=0.;global_seconds=0.
            if upper-lower>1e-4 and max(point_loss.max(),average.max())-lower>min(runner.c['raw_gap_thresholds']):
                if evaluator is None:
                    build_start=time.perf_counter()
                    evaluator=mixed.NeuralLogisticBound(models,origin,budget,constraints,1.,seed=identity['seed'])
                    build_seconds=time.perf_counter()-build_start
                evaluator.time_limit=runner.c['global_seconds_final'] if t==runner.c['rounds'] else runner.c['global_seconds_per_checkpoint']
                response,globalinfo=evaluator.solve(wb,response,local)
                global_seconds=globalinfo['solver_seconds']
            else:globalinfo=dict(skipped='independent bounds already tight or both gaps below smallest threshold')
            elapsed+=time.perf_counter()-started
            record=dict(round=t,played_loss=float(np.mean(np.sum(weights[:t]*losses[:t],axis=1))),
                diagnostic_elapsed_seconds=elapsed,global_solver_seconds=global_seconds,global_build_seconds=build_seconds,
                local_evaluation=localinfo,global_evaluation=globalinfo,diagnostic_version=DIAGNOSTIC_VERSION)
            aux=dict(average_point=xb,average_weights=wb,dual_response=response.copy(),evaluation_response=response.copy(),
                average_response_losses=average,point_losses=point_loss,component_loss_lower_bounds=component)
            add(record,aux)
            rows=persist()  # Includes revisions to all earlier dependent checkpoints.
        rows=persist()
        tmp=folder/'gaps.csv.tmp';pd.DataFrame(rows).to_csv(tmp,index=False);os.replace(tmp,folder/'gaps.csv')
        return rows,elapsed
    finally:
        if evaluator is not None:evaluator.close()
