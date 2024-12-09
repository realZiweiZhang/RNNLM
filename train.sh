#!/bin/bash 
#SBATCH --job-name=zh-LSTM-Word-Small-dp      # Job name Word Char
#SBATCH --mem=20G                      # Job memory request
#SBATCH --gres=gpu:1                    # Number of requested GPU(s) 
#SBATCH --time=16:00:00                   # Time limit days-hrs:min:sec
#SBATCH --error=zh-LSTM-Word-Small-dp.err
#SBATCH --output=zh-LSTM-Word-Small-dp.out

# conda activate /home/ziweiz1/gokhale_user/ziwei/ada_env/bw_env
CUDA_VISIBLE_DEVICES=0 python main.py \
    --use_gpu True \
    --use_words 1 \
    --use_chars 0 \
    --en_dataset False \
    --highway_layers 1 \
    --rnn_size 300 \
    --save_path 'checkpoint/zh-LSTM-Word-Small-dp' \
    --kernels '[1,2,3,4,5,6]'\
    --feature_maps '[25,50,75,100,125,150]' \