import torch
import torch.nn as nn


from dataloader import wsjDataset
from charCNN import charCNN
from highwaymlp import highwayMLP
from lstm import rnnLM

import argparse


def main(*args):
    if args.use_gpu:
        print('using CUDA on GPU to train the model')
    else:
        print('using CPU to train the model')

    # dataloader
    train_dataset = wsjDataset('train')
    word_vocab_size = train_dataset.get_word_vocab_size
    char_vocab_size = train_dataset.get_char_vocab_size
    max_word_l = train_dataset.get_max_word_l
    # TODO finish all cases determined by hsm
    assert hsm == 0, 'not implemented error'
    
    # initialization
    charcnn = charCNN()  
    highwaymlp = highwayMLP()
    rnnlm = rnnLM(args.rnn_size,
                  args.num_layers,
                  args.dropout,
                  word_vocab_size,
                  args.word_vec_size,
                  char_vocab_size,
                  args.char_vec_size,
                  args.feature_maps,
                  args.kernels,
                  args.get_max_word_l,
                  args.use_words,
                  args.use_chars,
                  args.batch_norm,
                  args.highway_layers,
                  args.hsm
                  )
    
    # TODO checkpoint
    
    # optimization
    criterion = nn.ClassNLLCriterion()


    if args.use_gpu:
        rnnlm.cuda()
        
    for i in range(args.max.epochs):
        
    
    
if __name__ == '__main__':  
    parser = argparse.ArgumentParser(desciption="Implementation for lstm-char-cnn")

    parser.add_argument('--data_dir',type=str, default="data/")
    parser.add_argument('--use_words',type=int, default=0)
    parser.add_argument('--use_chars',type=int, default=1)
    parser.add_argument('--highway_layers',type=int, default=2)
    parser.add_argument('--word_vec_size',type=int, default=650)
    parser.add_argument('--char_vec_size',type=int, default=15)
    parser.add_argument('--feature_maps',type=str, default='[50,100,150,200,200,200,200]')
    parser.add_argument('--kernels',type=str, default='[1,2,3,4,5,6,7]')
    parser.add_argument('--num_layers',type=int, default=2)
    parser.add_argument('--dropout',type=float, default=0.5)

    #optimization settings
    parser.add_argument('--hsm',type=int, default=0)
    parser.add_argument('--learning_rate',type=int, default=1)
    parser.add_argument('--learning_rate_decay',type=float, default=0.5)
    parser.add_argument('--decay_when',type=float, default=1)
    parser.add_argument('--param_init',type=float, default=0.05)
    parser.add_argument('--batch_norm',type=bool, default=False)
    parser.add_argument('--seq_length',type=int, default=35)
    parser.add_argument('--batch_size',type=int, default=20)
    parser.add_argument('--max_epochs',type=int, default=25)
    parser.add_argument('--max_grad_norm',type=int, default=5)
    parser.add_argument('--max_word_l',type=int, default=65)

    #running settings
    parser.add_argument('--seed',type=int, default=3435)
    parser.add_argument('--use_gpu',type=bool, default=True)
    
    args = parser.parse_args()

