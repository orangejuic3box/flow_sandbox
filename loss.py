'''
- wassertein1(earth movers distance) distance (torch cfm)
 W_2 squares distances, moving a small amount of mass very far away creates a massive penalty. W_2 prefers moving many points a short distance rather than moving a single point a long distance, whereas W_1 treats both equally if the total distance is the same

- energy distance
- mmd

All the metric and loss functions we are going to track during training/test

For ALL loss funcs:
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the source minibatch
'''
from torchcfm.optimal_transport import wasserstein

def was_1(x0, x1):
    '''
    Wraps the wassertein func from torchcfm
    
    Measures the minimum "work" or cost required to transform one probability distribution into another
    
    Weights distance LINEARLY
    '''
    return wasserstein(x0, x1, power=1)

def wass_2(x0, x1):
    '''
    Wraps the wassertein func from torchcfm, default power is 2
    
    Measures the minimum "work" or cost required to transform one probability distribution into another

    Weights distance QUADRATICALLY ^2 and then SQUAREROOT total
    '''
    return wasserstein(x0, x1)

def energy_distance(x0, x1, x_w = None, y_w = None):
    '''
    Evaluates whether two probability distributions are identical. The energy distance is 0 if and only if the distributions are the same, and greater than 0 if they differ.
    '''
    nx = get_backend(x0, x1)
   
    x_w = nx.full((x0.shape[0], ), 1/x0.shape[0]) if x_w is None else x_w / x_w.sum()
    y_w = nx.full((x1.shape[0], ), 1/x1.shape[0]) if y_w is None else y_w / y_w.sum()
    
    xy=nx.dot(x_w, pot.utils.euclidean_distances(x0, x1, squared=False) @ y_w)
    xx=nx.dot(x_w, pot.utils.euclidean_distances(x0, x0, squared=False) @ x_w)
    yy=nx.dot(y_w, pot.utils.euclidean_distances(x1, x1, squared=False) @ y_w)
    
    return 2*xy-xx-yy

def mmd_distance(x0, x1, gamma):
    '''
    Maximum Mean Discrepancy 

    A statistical distance measure used to determine whether two different sets of samples come from the same probability distribution
    '''
    xx = rbf_kernel(x0, x0, gamma)
    xy = rbf_kernel(x0, x1, gamma)
    yy = rbf_kernel(x1, x1, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()

