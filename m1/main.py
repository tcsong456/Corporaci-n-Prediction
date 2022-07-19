import argparse
from m1.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='data/processed')
    parser.add_argument('--test_size',type=float,default=0.25)
    parser.add_argument('--num_decode_steps',type=int,default=16)
    parser.add_argument('--use_bilstm',action='store_true',help='if bilstm model used info feature build')
    parser.add_argument('--residual_channel',type=int,default=32)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=0.005)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--decay',type=float,default=0.9)
    parser.add_argument('--optim',type=str,default='adam')
    parser.add_argument('--eval_interval_epoch',type=int,default=1)
    parser.add_argument('--early_stopping_epochs',type=int,default=3)
    parser.add_argument('--checkpoint_path',type=str,default='m1/checkpoint')
    parser.add_argument('--warm_start',action='store_true')
    parser.add_argument('--receiptive_field',type=int,default=9)
    args = parser.parse_args()
    
    trainer = Trainer(args)
    trainer.train()
#%%