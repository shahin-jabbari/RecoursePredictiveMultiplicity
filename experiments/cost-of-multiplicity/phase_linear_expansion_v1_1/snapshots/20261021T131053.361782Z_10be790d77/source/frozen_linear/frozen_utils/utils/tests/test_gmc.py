import unittest
from types import SimpleNamespace
import numpy as np
import torch
from utils import recourse_core as rc
from utils import recourse_gmc_mixed as gm
from utils import recourse_gmc_treediversity as td


class GlobalLogisticBounds(unittest.TestCase):
    def test_linear_bounds_contain_analytic_minimum(self):
        m=rc.LogisticRegressionModel(2,42)
        with torch.no_grad():m.linear.weight[:]=torch.tensor([[1.,-2.]]);m.linear.bias[:]=.3
        models=rc.ModelSet([m]);x=np.array([.1,.2]);b=.8
        c=rc.Constraints(np.full(2,-np.inf),np.full(2,np.inf),[],'test')
        scale=gm.common_normalizer([models],x,b,c)
        local=gm.LocalLogisticOracle(models,x,b,c,scale)
        point,_=local.solve(np.array([1.]),x)
        evaluator=gm.NeuralLogisticBound(models,x,b,c,scale,time_limit=1.)
        try:_,info=evaluator.solve(np.array([1.]),point,local)
        finally:evaluator.close()
        optimum=np.logaddexp(0,-(np.dot([1.,-2.],x)+float(m.linear.bias)+b*np.sqrt(5)))/scale
        self.assertLessEqual(info['lower_bound'],optimum+2e-6)
        self.assertGreaterEqual(info['upper_bound'],optimum-2e-6)

    def test_nonconvex_relu_bounds_contain_known_optimum(self):
        m=rc.NeuralNetworkModelSingleLayer(1,2,42)
        with torch.no_grad():
            m.net[0].weight[:]=torch.tensor([[1.],[-1.]])
            m.net[0].bias.zero_();m.net[2].weight[:]=1.;m.net[2].bias.zero_()
        models=rc.ModelSet([m]);x=np.zeros(1)
        c=rc.Constraints(np.array([-np.inf]),np.array([np.inf]),[],'test')
        scale=gm.common_normalizer([models],x,1.,c)
        local=gm.LocalLogisticOracle(models,x,1.,c,scale)
        evaluator=gm.NeuralLogisticBound(models,x,1.,c,scale,time_limit=1.)
        try:p,info=evaluator.solve(np.ones(1),x,local)
        finally:evaluator.close()
        optimum=np.logaddexp(0,-1.)/scale
        self.assertLessEqual(info['lower_bound'],optimum+2e-6)
        self.assertGreaterEqual(info['upper_bound'],optimum-2e-6)
        self.assertLess(info['upper_bound']-optimum,1e-4)
        self.assertLessEqual(abs(p[0]),1.+1e-8)

    def test_interval_bounds_cover_masked_relu_logits(self):
        m=rc.FeatureMaskedModel(rc.NeuralNetworkModelDoubleLayers(3,4,3,42),[1,0,1])
        models=rc.ModelSet([m]);x=np.array([.2,-.3,.7]);radius=1.
        c=rc.Constraints(np.array([-np.inf,x[1],-np.inf]),np.array([np.inf,x[1],np.inf]),[1],'test')
        lo,hi=gm.layer_bounds(m,x,radius,c)[-1]
        for angle in np.linspace(0,2*np.pi,100):
            p=x+np.array([np.cos(angle),0,np.sin(angle)])
            z=m.double().logits(torch.tensor(p)).item()
            self.assertGreaterEqual(z,float(lo[0])-1e-7);self.assertLessEqual(z,float(hi[0])+1e-7)


class TreeCoordinates(unittest.TestCase):
    def test_subset_splits_use_full_feature_coordinates(self):
        rng=np.random.default_rng(42);X=rng.normal(size=(200,4));y=(X[:,3]+X[:,1]>0).astype(int)
        ctx=SimpleNamespace(X_train=X[:150],y_train=y[:150],X_validation=X[150:],columns=['a','b','c','d'])
        m=td.fit_subset(ctx,np.array([0,1,0,1]),42,3)
        used=set(m.trees[0].feature[m.trees[0].left!=-1])
        self.assertTrue(used.issubset({1,3}));self.assertEqual(m.dimension,4)


if __name__=='__main__':unittest.main()
