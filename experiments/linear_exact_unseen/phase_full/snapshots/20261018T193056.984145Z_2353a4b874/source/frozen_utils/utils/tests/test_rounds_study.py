import unittest
import numpy as np
from utils import recourse_rounds_study as s, recourse_core as rc


class RoundsStudy(unittest.TestCase):
    def test_modes_change_only_sampling(self):
        base=s.settings()
        for mode in ['intermediate','full']:
            other=s.settings(mode)
            for key in set(base)-{'mode','datasets','seeds','max_instances'}:
                self.assertEqual(base[key],other[key],key)
        self.assertEqual(base['max_rounds'],200)
        self.assertEqual(base['primary_threshold'],.005)

    def test_nested_architecture_sets(self):
        previous=set()
        for n in [3,6,12,24]:
            indices=s.nested_indices('mixed_logistic',24,n)
            self.assertEqual(len(indices),n)
            self.assertTrue(previous<=set(indices));previous=set(indices)
            for left in [0,8,16]:self.assertEqual(sum(left<=x<left+8 for x in indices),n//3)

    def test_shared_majority_cohort_includes_ties(self):
        a=np.array([[.8,.2],[.2,.8],[.2,.2],[.8,.8]])
        b=np.array([[.8,.2,.2,.8],[.2,.8,.8,.8],[.2,.2,.8,.8],[.2,.8,.2,.8]])
        ids,common,_=s.paired_positions([a,b],'majority',5,42)
        self.assertEqual(set(ids),{0,2})
        self.assertEqual(common.tolist(),[True,False,True,False])

    def test_no_certification_from_small_lower_bound(self):
        self.assertEqual(s.bound_status(.001,.1,.005),'unresolved_bound_interval')
        self.assertEqual(s.bound_status(.006,.1,.005),'certified_above_threshold')
        self.assertEqual(s.bound_status(0,.004,.005),'certified_below_threshold')
        gate=s.GapStop(.005,2)
        self.assertFalse(gate.update(0,.004))
        self.assertFalse(gate.update(0,.02))
        self.assertFalse(gate.update(0,.004))
        self.assertTrue(gate.update(0,.004))
        self.assertFalse(gate.update(0,.004,False))

    def test_crossing_not_same_as_sustained_and_no_hit_is_missing(self):
        rows=[dict(round=t,gap_lower_bound=0,gap_upper_bound=u) for t,u in [(1,.003),(5,.008),(10,.002),(15,.002)]]
        r=s.threshold_summary(rows,.005,15)
        self.assertEqual(r['first_certified_round'],1)
        self.assertEqual(r['first_confirmed_round'],15)
        self.assertEqual(r['sustained_certified_round'],10)
        self.assertTrue(r['recrossed_after_first'])
        missed=s.threshold_summary(rows,.001,15)
        self.assertIsNone(missed['first_certified_round'])
        self.assertEqual(missed['final_status'],'unresolved_bound_interval')
        censored=s.threshold_summary(rows,.005,200)
        self.assertIsNone(censored['sustained_certified_round'])

    def test_early_stop_uses_upper_bound_and_fixed_horizon_eta(self):
        models=rc.ModelSet([rc.LogisticRegressionModel(2,42),rc.LogisticRegressionModel(2,43)])
        class Oracle:
            def losses(self,x):return np.array([.1,.1])
            def solve(self,w,start):return np.zeros(2),dict(objective=.1,optimizer_success=True)
        class Bound:
            def __init__(self,lb):self.lb=lb;self.closed=False
            def solve(self,w,x,local):return x,dict(lower_bound=self.lb,upper_bound=.1,bound_consistent=True)
            def close(self):self.closed=True
        c=s.settings(families=['convex_logistic'],model_counts=[2],max_rounds=10,gap_every=2,stop_on_gap=True)
        solver=Bound(.099)
        rows,trace,reason=s.trajectory(models,[0,0],1,'mixed_logistic',1,c,oracle=Oracle(),global_evaluator=solver)
        self.assertEqual(reason,'confirmed_gap_threshold');self.assertEqual(len(trace['points']),2)
        self.assertAlmostEqual(trace['eta'],np.sqrt(np.log(2)/10))
        self.assertTrue(solver.closed)
        rows,trace,reason=s.trajectory(models,[0,0],1,'mixed_logistic',1,c,oracle=Oracle(),global_evaluator=Bound(0))
        self.assertEqual(reason,'max_rounds');self.assertEqual(len(trace['points']),10)
        self.assertEqual(rows[-1]['primary_status'],'unresolved_bound_interval')


if __name__=='__main__':unittest.main()
