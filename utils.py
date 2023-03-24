import torch
import numpy as np
from sklearn.neighbors import KernelDensity

def wd(layers: list()):
    layers = [layers[0]]
    layers = np.array([layer*np.sqrt(depth+1) for depth, layer in enumerate(layers)]).flatten().reshape(-1, 1)
    res = KernelDensity(kernel="gaussian", bandwidth=0.4).fit(layers).sample([3])
    # return torch.from_numpy(np.array(res, dtype="float32"))
    return torch.from_numpy(np.array(np.array([1]), dtype="float32"))
    # return 1 - len(layers)/10 if len(layers)>0 else 1