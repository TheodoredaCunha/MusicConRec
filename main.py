import sys
from train import train
from eval import eval
from split_dataset import split_dataset
from inference import inference
# from scripts.train_sagemaker import train_sm

def info():
    print('info')

if __name__ == '__main__':
    mode = sys.argv[1]

    if mode == 'train': # train model
        train()
    # elif mode == 'trainsm': # Sagemaker training script is excluded from the GitHub
    #     train_sm()
    elif mode == 'eval': # evaluate model
        eval()
    elif mode == 'splitdata': # create training and validation split from MusicBench dataset (made to be reproducible using seed 42)
        split_dataset()
    elif mode == 'inference': # run inference on a sample audio file and print the embedding
        inference()
    else:
        info()
