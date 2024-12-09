# pip install -U conllu datasets
from tqdm import tqdm
import torch
import os

import transformers
from datasets import load_dataset, get_dataset_config_names
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dataloader import word_and_char_padding, create_y_label
# get ud dataset
name = "universal_dependencies"
ud_config = get_dataset_config_names(name)

non_en_opt = 'zh_gsdsimp'

special_tokens = {
    'UNK': '|',
    'START': '{',
    'END': '}',
}
# ud_data_train = load_dataset(name, non_en_opt, split="train")

class udDataset(Dataset):
    def __init__(self,split,vocab_dict=None):
        if split == 'train':
            self.word2idx = {0:0, 'UNK':1 }
            self.idx2word = {0:0, 1:'UNK'}
            
            self.idx2char = {0:0, 1:special_tokens['START'], 2:special_tokens['END'],3:special_tokens['UNK']}
            self.char2idx = {0:0, special_tokens['START']:1, special_tokens['END']:2,special_tokens['UNK']:3}
        
        else:
            self.word2idx = vocab_dict['word2idx']
            self.idx2word = vocab_dict['idx2word']
            
            self.idx2char = vocab_dict['idx2char']
            self.char2idx = vocab_dict['char2idx']
            
        self.lines = load_dataset(name, non_en_opt, split=split)
        self.split_dict = self.lines2word(split)
        
        self.max_word_lenth = self.split_dict['max_word_lenth']
        self.line_dict = self.split_dict['line_dict']
    
    def __len__(self):
        return len(self.line_dict)
    
    def __getitem__(self,idx):
        return self.line_dict[idx]
          
    def lines2word(self,split):
        num_word = 0
        line_dict = {}
        
        lines = self.lines
        max_word_length = 0
        max_line_length = 0
        for i, line in tqdm(enumerate(lines),total=len(lines)):
            line_word_idx = []
            line_chars = []
            
            words = line['tokens']
            # Only words containing characters and numbers are retained, and special symbols are eliminated, e.g. -
            # words = re.findall(r'\b\w+\b', line)
            # print('sentence being processing:\n',line)
        
            word_length = len(words)
            char_length = []
            max_line_length = max(len(words),max_line_length)
            for word in words:
                num_word += 1
                max_word_length = max(len(word), max_word_length)
                
                #TODO add spectial symbors into word dict
                if word not in self.word2idx:
                    if split == 'train':
                        self.idx2word[len(self.idx2word)] = word 
                        self.word2idx[word] = len(self.idx2word)-1
                        add_word = word
                    else:
                        add_word = 'UNK'
                else:
                    add_word = word
                # output_word_idx.append(word2idx[add_word])
                line_word_idx.append(self.word2idx[add_word])
                
                char_length.append(len(word))
                
                char_idx = self.word2char(word,split) # chars for each word
                line_chars.append(char_idx) # all chars for each line
                # output_chars.append(char_idx)
            y_labels = create_y_label(line_word_idx)
            assert len(y_labels) == word_length
            line_dict[i] = {'word_idx': line_word_idx, 
                            'char_idx': line_chars,
                            'y': y_labels,
                            'word_length':word_length, 
                            'char_length': char_length}
        
        line_dict = word_and_char_padding(line_dict,max_line_length, max_word_length)    
        output_dict = {
                    'max_word_lenth':max_word_length,
                    'max_char_length':max_line_length,
                    'line_dict':line_dict}
        return output_dict
    
    def word2char(self, word, split):
        chars = {1:self.char2idx[special_tokens['START']]}
        for char in word:
            if char not in self.char2idx:
                if split == 'train':
                    length = len(self.idx2char)
                    self.idx2char[length] = char
                    self.char2idx[char] = length-1
                    add_char = char
                else:
                    add_char = special_tokens['UNK']
            else:
                add_char = char
            chars[len(chars)+1] = self.char2idx[add_char]
        chars[len(chars)+1] = self.char2idx[special_tokens['END']]

        return list(chars.values())   

    @property
    def get_word_vocab_size(self):
        return len(self.idx2word)
    
    @property
    def get_char_vocab_size(self):
        return len(self.idx2char)
    
    @property
    def get_max_word_l(self):
        return self.max_word_lenth
    
    @property
    def get_vocab_dict(self):
        return {'word2idx':self.word2idx, 
                'idx2word':self.idx2word, 
                'idx2char':self.idx2char, 
                'char2idx':self.char2idx
                }


def get_non_en_dataloader(non_en_opt):
    pass
    
if __name__ == '__main__':
    # print(ud_data_train[2]['text'])
    print(ud_data_train[0]['tokens'])
    print(set(ud_data_train[0]['tokens']))
    print(len(list(set(ud_data_train[0]['tokens']))[0]))