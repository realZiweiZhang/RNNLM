import torch.nn as nn
import torch.nn.functional as F


class highwayMlp(nn.Module):
    def __init__(self, num_layers=1, dim):
        super().__init__()
        self.bias = -2

        self.linear_layers = nn.ModuleList()
        self.transform_gate = nn.ModuleList()
        
        for i in range(num_layers):
            self.linear_layers.append(nn.Linear(dim, dim))
            self.transform_gate.append(nn.Linear(dim, dim))
        
    def forward(self, x):
        
        