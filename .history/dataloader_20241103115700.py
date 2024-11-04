import os
import re

global_tokens = {
    
}


def get_lines_from_txt(split=None):
    assert split is not None, "split is not speficied"
    
    txt_path = os.path.join('data/wsj_'+str(split)+'.txt')
    print('load sentences from path:',txt_path)
    
    with open(txt_path, encoding="utf-8") as f:
        lines = f.readlines()
    
    print(lines[2])
    return lines
    
    
def line2word_process(line):
    # Only words containing characters and numbers are retained, and special symbols are eliminated, e.g. -
    words = re.findall(r'\b\w+\b', line)
    
    print('word being processing',)
    for word in words:
        print(word)
    

def word_process(): 

    pass

if __name__ == '__main__':
    train_lines = get_lines_from_txt('train')
    
    line2word_process(train_lines[1])
    