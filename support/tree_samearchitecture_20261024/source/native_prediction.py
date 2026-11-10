from common import *
from sklearn.tree import _tree
def native_saved_prediction(model,X):
 """Run preserved arrays through sklearn's native tree traversal without fitting."""
 outputs=[]
 for spec in model.trees:
  n=len(spec.left);nodes=np.zeros(n,dtype=_tree.NODE_DTYPE)
  for name,values in [('left_child',spec.left),('right_child',spec.right),('feature',spec.feature),('threshold',spec.threshold)]:nodes[name]=values
  nodes['n_node_samples']=1;nodes['weighted_n_node_samples']=1.
  values=np.stack([1-spec.probability,spec.probability],axis=1)[:,None,:].copy()
  tree=_tree.Tree(model.dimension,np.array([2],dtype=np.intp),1)
  tree.__setstate__({'max_depth':4,'node_count':n,'nodes':nodes,'values':values})
  pred=tree.predict(np.asarray(X,dtype=np.float32));outputs.append(pred[:,1]/pred.sum(axis=1))
 return np.mean(outputs,axis=0)
