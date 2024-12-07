import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dataloader import wsjDataset,collate_fn
from charCNN import charCNN
from highwaymlp import highwayMLP
from lstm import rnnLM

import argparse
from tqdm import tqdm
def train_one_epoch(model, dataloader, criterion, device, optimizer=None, train=False):
    if train:
        assert optimizer is not None
        model.train()
    else:
        model.eval()
    
    train_loss = 0
    ppl_over_dataset = []
    for data in tqdm(dataloader):
        words = torch.LongTensor(data['words'])
        chars = [torch.LongTensor(vec) for vec in data['chars']]
        chars = torch.nn.utils.rnn.pad_sequence(chars, batch_first=True, padding_value=0)

        y = data['y']
        words, chars = words.to(device), chars.to(device)
    
        outputs, prob = model(words, chars)

        loss, ppl = loss_func(criterion, prob, y)
        ppl_over_dataset.extend(ppl)
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss += loss
    
    print(ppl_over_dataset/len(dataloader))
    return train_loss/len(dataloader)

def calculate_ppl(batch_loss,y_len):
    return torch.exp(batch_loss/y_len)

def loss_func(criterion, outputs, y):
    # mask = torch.arange(outputs.shape[1]).unsqueeze(0).repeat(output.shape[0],1)
    mask =  torch.arange(outputs.shape[1])
    batch_loss = 0
    
    y_len = [len(y[i]) for i in range(outputs.shape[0])]
    rearr_outputs = [outputs[i][:y_len[i]] for i in range(outputs.shape[0])]
    rearr_outputs = torch.cat(rearr_outputs, dim=0)
    
    y_1d = []
    for item in y:
        y_1d.extend(item)
    y_1d  =torch.tensor(y_1d)
    
    # TODO PPL
    batch_loss = criterion(rearr_outputs,y_1d)
    batch_ppl = calculate_ppl(batch_loss, y_len)    
    return batch_loss, batch_ppl
     
def main(args):
    if args.use_gpu:
        print('using CUDA on GPU to train the model')
    else:
        print('using CPU to train the model')

    # dataloader
    train_dataset = wsjDataset('train')
    eval_dataset = wsjDataset('dev')
    test_dataset = wsjDataset('test')
    word_vocab_size = train_dataset.get_word_vocab_size
    char_vocab_size = train_dataset.get_char_vocab_size
    
    max_word_l = train_dataset.get_max_word_l
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,collate_fn=collate_fn, num_workers=0)
    eval_dataloader =  DataLoader(eval_dataset, batch_size=args.batch_size,collate_fn=collate_fn)
    test_dataloader =  DataLoader(test_dataset, batch_size=args.batch_size,collate_fn=collate_fn)
    
    # TODO finish all cases determined by hsm
    assert args.hsm == 0, 'not implemented error'
    print('start initialization')
    # initialization
    model = rnnLM(args.rnn_size,
                  args.num_layers,
                  args.dropout,
                  word_vocab_size,
                  args.word_vec_size,
                  char_vocab_size,
                  args.char_vec_size,
                  args.feature_maps,
                  args.kernels,
                  args.use_words,
                  args.use_chars,
                  args.batch_norm,
                  args.highway_layers,
                  args.hsm
                  )
    
    # TODO checkpoint
    
    # optimization
    lr = args.learning_rate
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()


    if args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # training stage
    min_eval_loss = 10e9
    epoch_loss = 0
    for epoch in tqdm(range(args.max_epochs)):
        print(f'start training')
        epoch_loss = train_one_epoch(model, train_dataloader, criterion, device, optimizer, train=True)
        print(f'training: epoch {epoch}, loss {epoch_loss}')
        
        # eval_stage
        with torch.no_grad():
            if epoch > args.max_epochs // 2:
                eval_epoch_loss = train_one_epoch(model, eval_dataloader, criterion, device, None, train=False)
                
                if eval_epoch_loss < min_eval_loss:
                    save_path = os.path.join(args.save_path,f"train_epoch{epoch:.2f}_{eval_epoch_loss:.2f}.pt")
                    
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr':lr
                    }
                    
                    torch.save(checkpoint, save_path)
            
        if args.learning_rate_decay > 0 and (epoch%2)==0:
            lr *= args.learning_rate_decay
        
    
    # test_stage
    
    
if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description="Implementation for lstm-char-cnn")

    parser.add_argument('--data_dir',type=str, default="data/")
    
    parser.add_argument('--rnn_size',type=int, default=650, help="size of LSTM internal state")
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
    parser.add_argument('--use_gpu',type=bool, default=False)
    parser.add_argument('--save_path',type=str, default='checkpoint/')
    
    args = parser.parse_args()
    
    main(args)

