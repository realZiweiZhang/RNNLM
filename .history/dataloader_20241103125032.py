import os
import re

global_tokens = {
    
}

word2idx = {}
idx2word = {}

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
    output_idx = []
    for line in lines:
        # Only words containing characters and numbers are retained, and special symbols are eliminated, e.g. -
        words = re.findall(r'\b\w+\b', line)
        print('sentence being processing',line)
        
        for word in words:
            num_word += 1
            print(word)
            
            #TODO add spectial symbors into word dict
            
            if word2idx.get([str(word)],True):
            idx2word[len(idx2word)+1] = word 
            word2idx[word] = len(idx)
            
            out
            

def word2char(word): 

    pass

if __name__ == '__main__':
    train_lines = get_lines_from_txt('train')
    
    line2word_process(train_lines[1])
    