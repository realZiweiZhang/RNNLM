f = open("data/wsj_train.txt")

lines = f.readlines()
print(lines[0])

def get_lines_from_txt(split):
    assert split !=