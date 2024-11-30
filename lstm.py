import torch.nn as nn

class Highway(nn.Module):
    def __init__(self,num_layer=2,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        
        self.linears = nn.ModuleList(nn.Linear(hidden_size,hidden_size) for _ in range(num_layer))
        self.g = nn.ReLU()
        
        self.transform_linears = nn.ModuleList(nn.Linear(hidden_size,hidden_size) for _ in range(num_layer))
        self.sigma = nn.Sigmoid()

    def forward(x)
        assert x.shape[-1] == self.hidden_size, "wrong feature size input to highway"
        
        for i in range(self.num_layer):
            h_x = self.g(self.linears[i](x))
            
            y_x = self.transform_linears[i](x)
            transform_gate = self.sigma(y_x)
            carry_gate = 1 - transform_gate
            
            x = transform_gate * h_x + carry_gate * x
        return x

class charCNN(nn.Module):
    

class RNN_LM(nn.Module):
    def __init__(self,
                 num_highway_layers=2,
                 char_cnn = None,
                 num_layer=2,
                 rnn_size=650,
                 char_vec_size=None,
                 word_vec_size=None,
                 use_chars=True,
                 use_words=False
                 ):
        super().__init__()
        self.highway_mlp = highway
        self.num_layer = num_layer
        
        self.use_chars = use_chars
        self.use_words = use_words
        
        self.input_size = 0
        
        assert char_vec_size is not None, 'undefined char vec size'
        assert word_vec_size is not None, 'undefined word vec size'
        
        if use_chars:
            self.input_size = char_vec_size
            self.char_cnn = None
            if use_words:
                self.input_size += word_vec_size
        else:
            self.input_size = word_vec_size
        
        self.highway_mlp = Highway(self.input_size, highway_layers)
        # self.rnn = nn.LSTM(input_size, hidden_size, num_layers,dropout=0.5)
        self.lstm_cells = nn.ModuleList(self.input_size if i == 0 else rnn_size, rnn)
        
        if self.highway_mlp is None:
            raise NotImplementedError("The 'highway function is not implemented")

        if self.char_cnn is None:
            raise NotImplementedError("The 'char_cnn function is not implemented")

        layers = []
        for i in range(self.num_layer):
            if i == 0:
                if use_chars:
                    layers.append(self.char_cnn)
                layers.append(self.highway_mlp)
            else:
                layers.append(nn.Dropout(p=0.5))
            
            
              
    def forward(x):
        """_summary_

        Args:
            x (_type_): output from the character-level CNN
        """
        for i 