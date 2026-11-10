"""Tests of pairing, feature constraints, and validation-only model selection."""
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from utils import recourse_core as rc
from utils import recourse_diversity as dv
from utils import recourse_results as rr
from utils import recourse_campaign as campaign
from utils import recourse_convergence as gap


class DiversityProtocol(unittest.TestCase):
    def test_whole_groups_and_actionable_omission(self):
        columns=['Attribute9_A91','Attribute9_A92','age','amount','duration']
        groups=dv.feature_groups(columns)
        self.assertEqual(groups['Attribute9'],[0,1])
        for seed in range(30):
            mask=dv.feature_mask(groups,[3,4],np.random.default_rng(seed),.5)
            self.assertEqual(mask[0],mask[1])
            self.assertTrue((mask[[3,4]]==0).any())
            self.assertTrue((mask[[3,4]]==1).any())
            self.assertGreater(mask.sum(),0)

    def test_accuracy_caliper_and_fallback(self):
        y=np.array([0,0,1,1])
        ref=np.array([[.1,.1],[.1,.7],[.9,.9],[.9,.9]])
        candidates=np.array([[.1],[.7],[.9],[.9]])
        chosen,acc=dv.select_matched(ref,candidates,np.array([[1,0]]),y,
            np.array([1.,.75]),.7,.02)
        self.assertEqual(chosen,[None,0])
        self.assertEqual(float(acc[0]),.75)

    def test_deployed_model_is_pinned_even_with_eligible_replacement(self):
        y=np.array([0,0,1,1])
        p=np.array([[.1,.2],[.1,.2],[.9,.8],[.9,.8]])
        chosen,_=dv.select_matched(p,p,np.array([[1,0],[0,1]]),y,
            np.array([1.,1.]),.95,.02,pinned_first=True)
        self.assertIsNone(chosen[0])
        self.assertIsNotNone(chosen[1])

    def test_masked_coordinates_have_no_prediction_or_gradient_effect(self):
        m=rc.FeatureMaskedModel(rc.LogisticRegressionModel(3,42),[1,0,1])
        x=torch.tensor([.1,.2,.3],requires_grad=True)
        p=m(x);p.backward()
        self.assertEqual(float(x.grad[1]),0.)
        x2=x.detach().clone();x2[1]=999
        torch.testing.assert_close(p.detach(),m(x2))
        a,b=rc.linear_parameters(m)
        torch.testing.assert_close(m.logits(x.detach()).reshape(()),(a@x.detach()+b).reshape(()))

    def test_reference_cohorts_nested_models_and_no_test_selection(self):
        X,y=rc.get_data_synthetic(2000)
        settings=dict(candidate_selection='reference',training_epochs=3,
            diversity_options=dict(candidate_pool_size=6,bank_capacity=4))
        token=rr._ACTIVE_SETTINGS.set(settings)
        try:
            old,xf=rc.train_and_select_candidates_diverse(X,y,2,0,0,max_instances=4,
                dataset_name='Synthetic',actionable_indices=[0,1],diversity=False)
            small,xs=rc.train_and_select_candidates_diverse(X,y,2,0,0,max_instances=4,
                dataset_name='Synthetic',actionable_indices=[0,1],diversity=True)
            large,xl=rc.train_and_select_candidates_diverse(X,y,4,0,0,max_instances=4,
                dataset_name='Synthetic',actionable_indices=[0,1],diversity=True)
            self.assertEqual(old.context.candidate_ids,small.context.candidate_ids)
            self.assertEqual(small.context.candidate_ids,large.context.candidate_ids)
            for a,b in zip(small,large):
                for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key])
            for record in large.metadata['model_matches']:
                self.assertLessEqual(abs(record['accuracy_difference']),.02+1e-12)
            ctx=rc.prepare_split(X,y,42,'Synthetic',[0,1]);ctx.y_test=1-ctx.y_test
            same,_=rc.train_and_select_candidates_diverse(X,y,4,0,0,max_instances=4,
                dataset_name='Synthetic',actionable_indices=[0,1],context=ctx,diversity=True)
            self.assertEqual(same.metadata['model_matches'],large.metadata['model_matches'])
        finally:rr._ACTIVE_SETTINGS.reset(token)

    def test_conditions_are_part_of_saved_profile(self):
        with patch.object(campaign,'DIVERSITY_ENABLED',False):
            off=campaign.profile_settings('quick',section='cost_validity')
        with patch.object(campaign,'DIVERSITY_ENABLED',True):
            on=campaign.profile_settings('quick',section='cost_validity')
            tree=campaign.profile_settings('quick',section='tree_duality_gap')
        self.assertNotEqual(campaign._hash(off),campaign._hash(on))
        self.assertTrue(tree['diversity'])
        self.assertEqual(off['folds'],on['folds'])

    def test_run_saves_new_diversity_source_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            run=rr.RunStore(directory,'source_snapshot_test',{})
            for name in ['recourse_diversity.py','recourse_diversity_study.py']:
                self.assertIn(name,run.manifest['source_sha256'])
                self.assertTrue((run.path/'source'/name).is_file())

    def test_shared_normalizer_keeps_loss_bound_and_pair_scale(self):
        models=[rc.LogisticRegressionModel(2,seed) for seed in [42,43]]
        base=gap.ConvexLogitOracle(models,np.zeros(2),2.)
        bound=base.normalizer*1.5
        shared=gap.ConvexLogitOracle(models,np.zeros(2),2.,normalizer=bound)
        np.testing.assert_allclose(shared.losses([.1,.2]),base.losses([.1,.2])/1.5)
        with self.assertRaises(ValueError):
            gap.ConvexLogitOracle(models,np.zeros(2),2.,normalizer=base.normalizer*.9)
        _,rows,_=gap.convex_trajectory(models,np.zeros(2),2.,T=4,every=2,normalizer=bound)
        self.assertTrue(all(row['normalizer']==bound for row in rows))


if __name__=='__main__':unittest.main()
