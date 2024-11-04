import os
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
    
    
def line_process() 

if __name__ == '__main__':
    get_lines_from_txt('train')
    
    