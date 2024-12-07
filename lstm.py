import torch.nn as nn
import torch
from charCNN import charCNN
from highwaymlp import highwayMLP

from einops import rearrange

class rnnLM(nn.Module):
    def __init__(self,
                 rnn_size=650,
                 num_layer=2,
                 dropout=0.5,
                 word_vocab_size=None,
                 word_vec_size=None,
                 char_vocab_size=None,
                 char_vec_size=None,
                 feature_maps=None,
                 kernels=None,
                 use_words=False,
                 use_chars=True,
                 batch_norm=False,
                 num_highway_layers=2,
                 hsm=0
                 ):
        super().__init__()
        self.num_layer = num_layer        
        self.use_chars = use_chars
        self.use_words = use_words
        
        self.input_size = 0
        self.rnn_size = rnn_size
        assert char_vec_size is not None, 'undefined char vec size'
        assert word_vec_size is not None, 'undefined word vec size'
        
        if use_chars:
            feature_maps, kernels = [int(dim) for dim in feature_maps[1:-1].split(',')],[int(dim) for dim in kernels[1:-1].split(',')]
            self.input_size = char_vec_size
            self.char_cnn = charCNN(feature_maps,kernels,self.input_size)
            self.input_size = sum(feature_maps)
            
            self.char_vec_layer = nn.Embedding(char_vocab_size+1, char_vec_size)
            if use_words:
                self.input_size += word_vec_size
                self.word_vec_layer = nn.Embedding(word_vocab_size, word_vec_size)
        else:
            self.input_size = word_vec_size
            self.word_vec_layer = nn.Embedding(word_vocab_size, word_vec_size)
        
        self.highway_mlp = highwayMLP(num_highway_layers, self.input_size)
        # self.rnn = nn.LSTM(input_size, hidden_size, num_layers,dropout=0.5)
        # self.lstm_cells = [nn.LSTMCell(self.input_size if i == 0 else rnn_size, rnn_size) for i in range(self.num_layer)]
        self.lstm = nn.LSTM(self.input_size, rnn_size, num_layers=self.num_layer, batch_first=True)
        self.drop_out= nn.Dropout(p=dropout)
        
        if self.highway_mlp is None:
            raise NotImplementedError("The 'highway function is not implemented")

        if self.char_cnn is None:
            raise NotImplementedError("The 'char_cnn function is not implemented")

        self.proj = nn.Linear(rnn_size,word_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
            
    def forward(self, word_indices=None, char_indices=None,lsm=0):
        """_summary_

        Args:
            x (_type_): output from the character-level CNN
        """
        if self.use_chars:
            assert char_indices is not None, "no char indices are passed to LSTM"
            # print(char_indices)
            char_embed = self.char_vec_layer(char_indices) # b length, char_len, emb
            
            char_embed = self.char_cnn(char_embed) # b word_length, embed
            if self.use_words:
                word_embed = self.word_vec_layer(word_indices) 
                char_embed = torch.cat((char_embed, word_embed),-1)
            x = char_embed
        else:
            x = self.word_vec_layer(word_indices)
        
        # initialize hx, cx
        # outputs = []
        hx,cx = torch.zeros(self.num_layer, x.shape[0],self.rnn_size),torch.zeros(self.num_layer, x.shape[0],self.rnn_size)
        # for i in range(self.num_layer):
        #     if i == 0:
        #        x = self.highway_mlp(x)
        #     hx, cx = self.lstm_cells[i](x,(hx,cx))
        
            # x = self.dropout(hx)
            # outputs.append((hx,cx))
        x = self.highway_mlp(x)
        # x = rearrange(x,'b l c -> l b c')
        outputs, (hn,cn) = self.lstm(x,(hx, cx))
        # print(outputs.shape)
        
        # outputs = rearrange(outputs, 'l b c -> b l c')
        # decoder
        if lsm > 0:
            return outputs, hn
        else:
            prob = self.softmax(self.proj(outputs))
            return outputs, prob