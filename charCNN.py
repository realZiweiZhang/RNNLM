import torch.nn as nn


class charCNN(nn.Module):
    def __init__(self,feature_maps,kernels,input_size):
        super().__init__()
        
        assert isinstance(feature_maps,list)
        assert isinstance(kernels,list)
        
        self.num_layer = len(feature_maps)
        layers = []
        for i in range(self.num_layer):
            layers.append(nn.Conv2d(1, feature_maps[i],(kernels[i],input_size)))

        self.cnn = nn.Sequential(*layers)
        
    def forward(self, x):
        """_summary_

        Args:
            x (b, length, input_size): _description_
        """
        x = x.unsqueeze(1)
        
        outputs = []
        for i in range(self.num_layer):
            
        