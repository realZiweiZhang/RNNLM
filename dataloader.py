import os
import re

import torch
from torch.utils.data import Dataset

special_tokens = {
    'UNK': '|',
    'START': '{',
    'END': '}',
}
# special_tokens.EOS = opt.EOS

word2idx = {}
idx2word = {}

idx2char = {1:special_tokens['START'], 2:special_tokens['END']} # global
char2idx = {special_tokens['START']:1, special_tokens['END']:2} # global
# char2idx[special_tokens['START']]=1
# char2idx[special_tokens['END']]=2

output_chars = {}

def get_lines_from_txt(split=None):
    assert split is not None, "split is not speficied"
    
    txt_path = os.path.join('data/wsj_'+str(split)+'.txt')
    print('load sentences from path:',txt_path)
    
    with open(txt_path, encoding="utf-8") as f:
        lines = f.readlines()
    
    print(lines[2])
    return lines
    
def lines2word(lines):
    num_word = 0
    output_dict = {}
    output_word_idx = [] # local
    output_chars = []
    
    for i, line in enumerate(lines):
        line_word_idx = []
        line_chars = []
        # Only words containing characters and numbers are retained, and special symbols are eliminated, e.g. -
        words = re.findall(r'\b\w+\b', line)
        print('sentence being processing:\n',line)
        
        for word in words:
            num_word += 1
            print(word)
            
            #TODO add spectial symbors into word dict
            
            if word not in word2idx:
                idx2word[len(idx2word)+1] = word 
                word2idx[word] = len(idx2word)
                
            output_word_idx.append(word2idx[word])
            line_word_idx.append(word2idx[word])
            
            chars = word2char(word)
            line_chars.append(chars)
            output_chars.append(chars)
        output_dict[i] = {'word_idx': line_word_idx, 'chars': line_chars}
    
    output_dict = {'all_word_idx':output_word_idx, 'all_chars': output_chars}
    return output_dict

def word2char(word): 
    chars = {1:char2idx[special_tokens['START']]}
    for char in word:
        if char not in char2idx.keys():
            idx2char[len(idx2char)+1] = char
            char2idx[char] = len(idx2char)

        chars[len(chars)+1] = char2idx[char]
    chars[len(chars)+1] = char2idx[special_tokens['END']]

    return chars
    
#TODO length
def max_length_truncation(chars, max_length):
    pass
    
def char_dataset_generation(split):
    split_lines = get_lines_from_txt(split)
    split_dict = lines2word(split_lines)
    
    return split_dict


class wsjDataset(Dataset):
    def __init__(self,split):
        self.split_dict = char_dataset_generation(split)
        
    def __len__(self):
        return len(split_dict.keys())-2
    
    def __getitem__(self,idx):
        line_dict = self.split_dict[int(idx)]
        
        words_idx = line_dict['word_idx']  
        chars_idx = line_dict['chars']
        return words_idx, chars_idx
    
    @property
    def get_word_vocab_size(self):
        return len(idx2word)
    
    @property
    def get_char_vocab_size(self):
        return len(idx2char)
    
    @property
    def get_max_word_l(self):
        raise NotImplementedError

if __name__ == '__main__':
    train_lines = get_lines_from_txt('train')
    print('size of word vocabulary %d', len(idx2word))
    print(lines2word([train_lines[1]]))
    
    