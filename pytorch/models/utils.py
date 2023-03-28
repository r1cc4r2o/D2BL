import torch
import numpy as np
from sklearn.neighbors import KernelDensity

def wd(current_layer):
    if current_layer.depth==0:
        return torch.from_numpy(np.array(np.array([1]), dtype="float32"))
    layers = np.array([layer.weight.detach().cpu().numpy() for layer in current_layer.layers]).flatten().reshape(-1, 1)
    mask = layers > 0
    layers = layers[mask].reshape(-1, 1)
    res = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(layers).sample([current_layer.out_features])
    return torch.from_numpy(np.array(res, dtype="float32"))
