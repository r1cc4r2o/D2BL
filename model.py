import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import wd

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(256, 512)
        self.l2 = LinW(in_features=512, out_features=512, depth=1)
        self.l3 = LinW(in_features=512, out_features=512, depth=2)
        self.l4 = nn.Linear(512, 10)
        self.gelu = nn.GELU()
        self.layers = [self.l2, self.l3]

    def forward(self, x):
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
    def __init__(self, in_features, out_features, depth):
        super(LinW, self).__init__(in_features=in_features, out_features=out_features)
        self.depth = depth

    def forward(self, input):
        weight_decay = wd([self])
        weight = self.weight * weight_decay
        return F.linear(input, weight, self.bias)