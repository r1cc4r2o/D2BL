import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 10)
        self.gelu = nn.GELU()

    def forward(self, x):
        repr = []
        x = self.flatten(x)
        x = self.gelu(self.l1(x))
        x = self.gelu(self.l2(x))
        x = self.gelu(self.l3(x))
        x = self.l4(x)
        return x