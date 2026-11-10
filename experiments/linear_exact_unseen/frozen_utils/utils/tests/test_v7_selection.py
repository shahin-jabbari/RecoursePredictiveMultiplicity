"""Eligibility, cache separation, and majority-cohort analysis regressions."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from utils import recourse_core as rc, recourse_campaign as campaign, recourse_results as rr
from utils import recourse_diversity as diversity
from utils import recourse_diversity_study as study


class Lookup(torch.nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.register_buffer('probabilities', torch.tensor(probabilities,dtype=torch.float32))

    def forward(self,x):
        return self.probabilities[x[...,0].long()]


def fixtures(matrix):
    matrix=np.asarray(matrix)
    ctx=SimpleNamespace(X_test=np.arange(len(matrix),dtype=np.float32)[:,None],
        test_ids=np.arange(len(matrix))+100,seed=42,candidate_ids=[],candidate_selection={},columns=['row'])
    return rc.ModelSet([Lookup(matrix[:,i]) for i in range(matrix.shape[1])],context=ctx)


class Selection(unittest.TestCase):
    def test_reference_and_at_least_half_are_distinct(self):
        m=fixtures([[.2,.9,.9,.9],[.9,.2,.2,.9],[.2,.2,.9,.9],[.5,.5,.5,.5]])
        rc.select_candidates(m,20,candidate_selection='reference',order='test')
        self.assertEqual(m.context.candidate_ids,[100,102])
        rc.select_candidates(m,20,candidate_selection='majority',order='test')
        self.assertEqual(m.context.candidate_ids,[101,102])
        self.assertEqual(m.context.candidate_selection['selected_rejection_counts'],[2,2])
        self.assertEqual(m.context.candidate_selection['selected_reference_rejected'],[False,True])

    def test_odd_set_rounds_up_and_threshold_equality_is_acceptance(self):
        m=fixtures([[.1,.1,.5,.8,.9],[.1,.1,.1,.8,.9]])
        rc.select_candidates(m,20,candidate_selection='majority',order='test')
        self.assertEqual(m.context.candidate_ids,[101])
        self.assertEqual(m.context.candidate_selection['minimum_rejections'],3)

    def test_single_model_rules_agree(self):
        m=fixtures([[.1],[.5],[.9]])
        for rule in ['reference','majority']:
            rc.select_candidates(m,10,candidate_selection=rule)
            self.assertEqual(m.context.candidate_ids,[100])

    def test_empty_and_short_cohorts_never_fill_with_ineligible_points(self):
        m=fixtures([[.9,.9],[.9,.9]])
        self.assertEqual(rc.select_candidates(m,5,candidate_selection='majority'),[])
        self.assertEqual(m.context.candidate_selection['status'],'no_eligible_factuals')
        m=fixtures([[.1,.8],[.9,.9]])
        rc.select_candidates(m,5,candidate_selection='majority')
        self.assertEqual(m.context.candidate_ids,[100])
        self.assertEqual(m.context.candidate_selection['eligible_instances'],1)

    def test_reference_override_and_reproducible_subsampling(self):
        m=fixtures([[.1,.9],[.1,.9],[.9,.1],[.9,.1]])
        rc.select_candidates(m,1,reference_model=m[1],candidate_selection='reference')
        first=list(m.context.candidate_ids)
        self.assertIn(first[0],[102,103])
        rc.select_candidates(m,1,reference_model=m[1],candidate_selection='reference')
        self.assertEqual(first,m.context.candidate_ids)

    def test_whole_forests_count_as_models(self):
        class Forest:
            trees=list(range(50))
            def __init__(self,p):self.p=np.asarray(p)
            def predict_probability(self,x):return self.p
        m=fixtures([[.9,.9],[.9,.9]])
        m[:]=[Forest([.9,.9]),Forest([.1,.9])]
        rc.select_candidates(m,10,candidate_selection='majority',order='test')
        self.assertEqual(m.context.candidate_ids,[100])
        self.assertEqual(m.context.candidate_selection['actual_models'],2)

    def test_invalid_rule_and_nonfinite_predictions_fail(self):
        m=fixtures([[.1,.9]])
        with self.assertRaises(ValueError):rc.select_candidates(m,candidate_selection='strict_majority')
        with self.assertRaises(ValueError):rc.select_candidates(fixtures([[np.nan,.1]]),candidate_selection='majority')

    def test_training_and_diversity_are_independent_of_cohort_rule(self):
        X,y=rc.get_data_synthetic(2000)
        settings=dict(training_epochs=3,diversity=True,
            diversity_options=dict(candidate_pool_size=6,bank_capacity=4))
        results=[]
        for rule in ['reference','majority']:
            token=rr._ACTIVE_SETTINGS.set(dict(settings,candidate_selection=rule))
            try:
                models,facts=rc.train_and_select_candidates_diverse(X,y,4,0,0,
                    max_instances=5,dataset_name='Synthetic',actionable_indices=[0,1])
                results.append(models)
                p=diversity.probabilities(models,np.stack([x.numpy() for x in facts]))
                if rule=='majority':self.assertTrue((2*(p<.5).sum(axis=1)>=len(models)).all())
                else:self.assertTrue((p[:,0]<.5).all())
                self.assertEqual(models.metadata['candidate_selection']['mode'],rule)
            finally:rr._ACTIVE_SETTINGS.reset(token)
        self.assertEqual(results[0].metadata['model_hashes'],results[1].metadata['model_hashes'])

    def test_saved_audit_and_separate_run_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            old=(rr.RESULTS_ROOT,rr.NOTEBOOK_VERSION,rc.RESULTS_DIR)
            try:
                rr.configure_notebook('V7',directory)
                paths=[]
                for rule in ['reference','majority']:
                    with patch.object(campaign,'CANDIDATE_SELECTION',rule):
                        settings=campaign.profile_settings('quick',section='cost_validity')
                        paths.append(rr.start_notebook_experiment('cost_validity',settings_factory=lambda *_:settings))
                    run=rr.RunStore(paths[-1],'selection',settings)
                    token=rr._ACTIVE_RUN.set(run)
                    try:rc.select_candidates(fixtures([[.9,.1],[.9,.9]]),10,candidate_selection=rule)
                    finally:rr._ACTIVE_RUN.reset(token)
                    audit=json.loads((run.path/'seed_42_fold_0/cohort_selection.json').read_text())
                    self.assertEqual(audit['mode'],rule)
                    with np.load(run.path/'seed_42_fold_0/cohort_selection.npz') as a:
                        self.assertEqual(a['rejection_counts'].tolist(),[1,0])
                self.assertNotEqual(paths[0],paths[1])
                self.assertIn('reference',paths[0].parts)
                self.assertIn('majority',paths[1].parts)
            finally:rr.RESULTS_ROOT,rr.NOTEBOOK_VERSION,rc.RESULTS_DIR=old

    def test_only_duality_gap_sections_replace_loss_convergence(self):
        self.assertNotIn('convergence',campaign.SECTIONS)
        self.assertNotIn('tree_convergence',campaign.SECTIONS)
        for s in ['duality_gap','mixed_duality_gap','tree_duality_gap']:
            self.assertIn(s,campaign.SECTIONS)
            self.assertTrue(campaign.jobs_for(s,'quick'))
        with self.assertRaises(ValueError):campaign.jobs_for('convergence','quick')

    def test_pairing_uses_common_individuals_not_different_cohort_means(self):
        rows=[]
        for enabled,items in [(False,[(1,1.),(2,0.)]),(True,[(2,1.),(3,0.)])]:
            for iid,valid in items:
                rows.append(dict(section='cost_validity',dataset='test',n_models=4,budget=1.,method='ORPM',
                    seed=42,instance_id=iid,diversity=enabled,valid=valid,max_loss=1-valid,cost=1.,
                    gap_upper_bound=None,candidate_selection='majority'))
        result=study.paired_summary(pd.DataFrame(rows))
        valid=result[result.metric.eq('valid')].iloc[0]
        self.assertEqual(valid.n_paired_individuals,1)
        self.assertEqual(valid.difference_mean,100.)

    def test_no_paired_overlap_produces_unavailable_estimate(self):
        rows=[dict(section='cost_validity',dataset='test',n_models=4,budget=1.,method='ORPM',seed=42,
            instance_id=int(enabled),diversity=enabled,valid=True,max_loss=0.,cost=1.,gap_upper_bound=None,
            candidate_selection='majority') for enabled in [False,True]]
        result=study.paired_summary(pd.DataFrame(rows))
        self.assertTrue((result.n_seeds==0).all())
        self.assertTrue(result.difference_mean.isna().all())

    def test_paired_report_handles_no_eligible_factuals(self):
        diagnostics=pd.DataFrame([dict(dataset='Synthetic',section='duality_gap',seed=42,
            diversity=enabled,n_models=3,n_instances=0) for enabled in [False,True]])
        outcomes=pd.DataFrame(columns=['section','dataset','n_models','budget','method'])
        curves=pd.DataFrame(columns=['dataset','seed','instance_id','round','diversity'])
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(study._read_jsonl(Path(directory)/'missing_checkpoints.jsonl'),[])
            with patch.object(study,'collect',return_value=(diagnostics,outcomes,curves,{})):
                report=study.report([],directory,False)
            self.assertTrue(report['summary'].empty)
            self.assertTrue((Path(directory)/'README.md').is_file())


if __name__=='__main__':unittest.main()
