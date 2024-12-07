import torch.nn as nn
import torch
from einops import rearrange
class charCNN(nn.Module):
    def __init__(self,feature_maps,kernels,input_size):
        super().__init__()
        
        assert isinstance(feature_maps,list)
        assert isinstance(kernels,list)
        
        self.num_layer = len(feature_maps)
        
        self.max_kernel = max(kernels)
        
        self.conv_layers = [nn.Conv1d(input_size, feature_maps[i],kernels[i],padding=0) for i in range(self.num_layer)]
        # self.pooling_layers = [nn.MaxPool1d(kernel_size=1) for i in range(self.num_layer)]
            
        # self.cnn = nn.Sequential(*layers)
        # print(self.cnn)
        
    def forward(self, x):
        """_summary_

        Args:
            x (b, length, input_size): _description_
        """
        bs = x.shape[0]
        x = rearrange(x, 'b wl cl d -> (b wl) d cl')
        
        outputs = []
        for i in range(self.num_layer):
            x_padded = x
            if self.max_kernel > x.shape[-1]:
                padding_length = self.max_kernel - x.shape[-1]
                x_padded = nn.functional.pad(x,(padding_length//2, padding_length-padding_length//2))
            # outputs.append(self.pooling_layers[i](self.conv_layers[i](x)).squeeze())

            x_conv = torch.tanh(self.conv_layers[i](x_padded))
            x_pooling = nn.MaxPool1d(kernel_size=x_conv.shape[-1])(x_conv)
            # print(x_pooling.shape)
            outputs.append(x_pooling.squeeze())
        
        outputs = torch.cat(outputs,dim=1)
        outputs = rearrange(outputs, '(b wl) d -> b wl d', b = bs)
        return outputs

if __name__ == '__main__':
    charcnn = charCNN([50,100,150,200,200,200,200], [1,2,3,4,5,6,7], 256)
    
    input_seq = torch.randn(16, 5, 256)
    output_feature = charcnn(input_seq)
    
    print(output_feature.shape)
    