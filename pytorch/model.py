import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import wd

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(784, 256)
        self.l2 = LinW(in_features=256, out_features=256, depth=0)
        self.l3 = LinW(in_features=256, out_features=256, depth=1, layers=[self.l2])
        self.l4 = nn.Linear(256, 10)
        self.gelu = nn.GELU()
        self.layers = [self.l2, self.l3]

    def forward(self, x):
        repr = []
        x = self.flatten(x)
        x = self.gelu(self.l1(x))
        repr.append(x.detach().cpu().numpy())
        x = self.gelu(self.l2(x, repr))
        repr.append(x.detach().cpu().numpy())
        x = self.gelu(self.l3(x, repr))
        x = self.l4(x)
        return x
    
    def __getitem__(self, idx):
        return self.layers[idx]
    
    def __len__(self):
        return len(self.layers)
    

class LinW(nn.Linear):
    def __init__(self, in_features, out_features, depth, layers=[]):
        super(LinW, self).__init__(in_features=in_features, out_features=out_features)
        self.depth = depth
        self.layers = layers[:self.depth] if len(layers)>0 else layers

    def forward(self, input, prev=[]):
        weight_decay = wd(prev)
        weight = self.weight * weight_decay.to('cuda:0')
        return F.linear(input, weight, self.bias)