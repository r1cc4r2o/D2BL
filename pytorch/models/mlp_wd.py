import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import wd

class MLPWD(nn.Module):
    def __init__(self):
        super(MLPWD, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(784, 256)
        self.l2 = LinW(in_features=256, out_features=256, depth=0, layers=[])
        self.l3 = LinW(in_features=256, out_features=256, depth=1, layers=[self.l2])
        self.l4 = nn.Linear(256, 10)
        self.gelu = nn.GELU()
        self.layers = [self.l2, self.l3]

    def forward(self, x):
        x = self.flatten(x)
        x = self.gelu(self.l1(x))
        x = self.gelu(self.l2(x))
        x = self.gelu(self.l3(x))
        x = self.l4(x)
        return x
    
    def __getitem__(self, idx):
        return self.layers[idx]
    
    def __len__(self):
        return len(self.layers)
    

class LinW(nn.Linear):
    def __init__(self, in_features, out_features, depth, layers):
        super(LinW, self).__init__(in_features=in_features, out_features=out_features)
        self.depth = depth
        self.layers = layers

    def forward(self, input):
        weight_decay = wd(self)
        weight = self.weight * weight_decay.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        return F.linear(input, weight, self.bias)