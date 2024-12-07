import torch.nn as nn
import torch.nn.functional as F


# class highwayMlp(nn.Module):
#     def __init__(self, num_layers=1, dim):
#         super().__init__()
#         self.bias = -2

#         self.linear_layers = nn.ModuleList()
#         self.transform_gate = nn.ModuleList()
        
#         for i in range(num_layers):
#             self.linear_layers.append(nn.Linear(dim, dim))
#             self.transform_gate.append(nn.Linear(dim, dim))
        
#     def forward(self, x):
        

class highwayMLP(nn.Module):
    def __init__(self,num_layer=2,hidden_size=650):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        
        self.linears = nn.ModuleList(nn.Linear(hidden_size,hidden_size) for _ in range(num_layer))
        self.g = nn.ReLU()
        
        self.transform_linears = nn.ModuleList(nn.Linear(hidden_size,hidden_size) for _ in range(num_layer))
        self.sigma = nn.Sigmoid()

    def forward(self,x):
        assert x.shape[-1] == self.hidden_size, "wrong feature size input to highway"
        
        for i in range(self.num_layer):
            h_x = self.g(self.linears[i](x))
            
            y_x = self.transform_linears[i](x)
            transform_gate = self.sigma(y_x)
            carry_gate = 1 - transform_gate
            
            x = transform_gate * h_x + carry_gate * x
        return x
   