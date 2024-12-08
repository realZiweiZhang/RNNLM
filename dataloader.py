import os
import re
from tqdm import tqdm
import torch
from torch.utils.data import Dataset,DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

special_tokens = {
    'UNK': '|',
    'START': '{',
    'END': '}',
}
# special_tokens.EOS = opt.EOS

word2idx = {0:0, 'UNK':1 }
idx2word = {0:0, 1:'UNK'}

idx2char = {0:0, 1:special_tokens['START'], 2:special_tokens['END'],3:special_tokens['UNK']} # global
char2idx = {0:0, special_tokens['START']:1, special_tokens['END']:2,special_tokens['UNK']:3} # global
# char2idx[special_tokens['START']]=1
# char2idx[special_tokens['END']]=2

output_chars = {}

def get_lines_from_txt(split=None):
    assert split is not None, "split is not speficied"
    
    txt_path = os.path.join('data/wsj_'+str(split)+'.txt')
    print('load sentences from path:',txt_path)
    
    with open(txt_path, encoding="utf-8") as f:
        lines = f.readlines()
    
    # print(lines[2])
    return lines
    
def lines2word(lines,split):
    num_word = 0
    line_dict = {}
    output_word_idx = [] # local
    output_chars = []
    
    max_word_length = 0
    max_line_length = 0
    for i, line in tqdm(enumerate(lines),total=len(lines)):
        line_word_idx = []
        line_chars = []
        # Only words containing characters and numbers are retained, and special symbols are eliminated, e.g. -
        words = re.findall(r'\b\w+\b', line)
        # print('sentence being processing:\n',line)
        
        word_length = len(words)
        char_length = []
        max_line_length = max(len(words),max_line_length)
        for word in words:
            num_word += 1
            max_word_length = max(len(word), max_word_length)
            
            #TODO add spectial symbors into word dict
            if word not in word2idx:
                if split == 'train':
                    idx2word[len(idx2word)] = word 
                    word2idx[word] = len(idx2word)-1
                    add_word = word
                else:
                    add_word = 'UNK'
            else:
                add_word = word
            output_word_idx.append(word2idx[add_word])
            line_word_idx.append(word2idx[add_word])
            
            char_length.append(len(word))
            
            char_idx = word2char(word,split) # chars for each word
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

def word_and_char_padding(line_dict,max_line_length, max_word_length):
    for i in range(len(line_dict)):
        line = line_dict[i]
        chars_padded_list = []
        words = line['word_idx']
        words_padded = words[:max_line_length]
        words_padded = words_padded + [0]*(max_line_length-len(words_padded))
        for chars in line_dict[i]['char_idx']:
            chars_padded = chars[:max_word_length]
            chars_padded = chars_padded + [0]*(max_word_length-len(chars))
            assert len(chars_padded) == max_word_length
            chars_padded_list.append(chars_padded)
            
        # assert len(chars_padded_list) == max_word_length
        assert len(words_padded) == max_line_length
        line_dict[i]['char_idx'] = chars_padded_list
        line_dict[i]['word_idx'] = words_padded
    return line_dict

def word2char(word,split): 
    chars = {1:char2idx[special_tokens['START']]}
    for char in word:
        if char not in char2idx:
            if split == 'train':
                length = len(idx2char)
                idx2char[length] = char
                char2idx[char] = length-1
                add_char = char
            else:
                add_char = special_tokens['UNK']
        else:
            add_char = char
        chars[len(chars)+1] = char2idx[add_char]
    chars[len(chars)+1] = char2idx[special_tokens['END']]

    return list(chars.values())    

#TODO length
def max_length_truncation(chars, max_length):
    pass
    
def char_dataset_generation(split):
    split_lines = get_lines_from_txt(split)
    split_dict = lines2word(split_lines,split)
    
    return split_dict

def create_y_label(words):
    y = [0]* len(words)
    
    y[:-1] = words[1:]
    y[-1] = words[0]
    
    return y

class wsjDataset(Dataset):
    def __init__(self,split):
        self.split_dict = char_dataset_generation(split)
        self.max_word_lenth = self.split_dict['max_word_lenth']
        self.line_dict = self.split_dict['line_dict']
        
        # TODO max_line_length
        
    def __len__(self):
        return len(self.line_dict)
    
    def __getitem__(self,idx):
        return self.line_dict[idx]
        # words_idx = self.line_dict[idx]['word_idx']  
        # chars_idx = self.line_dict[idx]['char_idx'] # each char starts with 'start' and ends with 'end' or 'padding_zero'
        
        # y = self.line_dict[idx]['y']
        # # y = self.create_y_label(words_idx)
        # return words_idx, chars_idx, y, self.line_dict[idx]['word_length'],self.line_dict[idx]['char_length']
    

    
    @property
    def get_word_vocab_size(self):
        return len(idx2word)
    
    @property
    def get_char_vocab_size(self):
        return len(idx2char)
    
    @property
    def get_max_word_l(self):
        return self.max_word_lenth

def collate_fn(data):
    words = []
    chars = []
    y = []
    word_len = []
    char_len = [] 
    
    for item_dict in data:
        # word, char, y, word_length, char_length = item
        words.append(item_dict['word_idx'])
        chars.append(item_dict['char_idx'])
        y.append(item_dict['y'])
        word_len.append(item_dict['word_length'])
        char_len.append(item_dict['char_length'])
    
    # padded_sequences = torch.nn.utils.rnn.pad_sequence(words, batch_first=True, padding_value=0)
    return {'words': words,
            'chars':chars, 
            'y':y,
            'word_len':word_len,
            'char_len':char_len}


if __name__ == '__main__':
    train_lines = get_lines_from_txt('train')
    
    print('size of word vocabulary %d', len(idx2word))
    print(lines2word([train_lines[1]]))
    
    dataset = wsjDataset('train')
    
    print(char2idx)
    assert 0

    dataloder = DataLoader(dataset, batch_size=2,collate_fn=collate_fn)
    
    for data in dataloder:
        print(data['words'])
        print(data['chars'])
        assert 0