"""COMPAS appendix restrictions must never discard fixed-feature constraints."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from utils import recourse_core as rc


class CompasConstraints(unittest.TestCase):
    def setUp(self):
        self.origin = torch.tensor([0., 0., 0.])
        context = SimpleNamespace(
            name='COMPAS', columns=['age', 'priors_count', 'sex_Male'],
            actionable_indices=[0, 1, 2], actionability_policy='compas_appendix',
            scaler=SimpleNamespace(scale_=np.array([10., 1., 1.])))
        model = rc.LogisticRegressionModel(3, seed=42)
        rc.set_model_weights(model, torch.tensor([0., 1., 0., -.5]))
        self.models = rc.ModelSet([model], context=context)

    def test_explicit_immutable_priors_remain_fixed_in_optimizer(self):
        point = rc.oracle_gradient_descent(
            [1.], self.models, self.origin, 1., rc.recourse_loss,
            immutable_indices=[1])
        self.assertEqual(float(point[1]), float(self.origin[1]))

    def test_policy_unions_with_explicit_immutables(self):
        constraints = rc.constraints_for(self.models, self.origin, [1, 2, 1])
        self.assertEqual(constraints.immutable, [1, 2])
        np.testing.assert_array_equal(constraints.lower[[1, 2]], [0., 0.])
        np.testing.assert_array_equal(constraints.upper[[1, 2]], [0., 0.])

    def test_context_immutables_remain_fixed(self):
        self.models.context.actionable_indices = [0, 2]
        constraints = rc.constraints_for(self.models, self.origin)
        self.assertEqual(constraints.immutable, [1, 2])
        point = rc.oracle_gradient_descent(
            [1.], self.models, self.origin, 1., rc.recourse_loss)
        self.assertEqual(float(point[1]), 0.)

    def test_explicitly_immutable_age_overrides_age_increase_permission(self):
        constraints = rc.constraints_for(self.models, self.origin, [0])
        self.assertEqual(constraints.immutable, [0, 2])
        self.assertEqual(constraints.lower[0], 0.)
        self.assertEqual(constraints.upper[0], 0.)

    def test_actionable_age_keeps_original_five_year_bound(self):
        constraints = rc.constraints_for(self.models, self.origin)
        self.assertEqual(constraints.immutable, [2])
        self.assertEqual(constraints.lower[0], 0.)
        self.assertEqual(constraints.upper[0], .5)  # 5 original years / training scale.
        projected = rc.project_l2_ball(torch.tensor([1., 1., 1.]), self.origin, 10.,
                                       constraints=constraints)
        torch.testing.assert_close(projected, torch.tensor([.5, 1., 0.]))

    def test_ordinary_policy_still_unions_explicit_and_context_immutables(self):
        self.models.context.actionable_indices = [1]
        constraints = rc.constraints_for(self.models, self.origin, [1], policy='immutable')
        self.assertEqual(constraints.immutable, [0, 1, 2])
        np.testing.assert_array_equal(constraints.lower, self.origin.numpy())
        np.testing.assert_array_equal(constraints.upper, self.origin.numpy())

    def test_numpy_immutable_indices_are_accepted(self):
        constraints = rc.constraints_for(self.models, self.origin, np.array([0, 1]))
        self.assertEqual(constraints.immutable, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
