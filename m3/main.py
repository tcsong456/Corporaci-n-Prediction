import argparse
import warnings
import sys
sys.path.append('.')
warnings.filterwarnings(action='ignore')
from m3.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_window',type=int,default=180)
    parser.add_argument('--predict_window',type=int,default=16)
    parser.add_argument('--backoffset',type=int,default=0)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--hidden_size',type=int,default=512,)
    parser.add_argument('--encoder_layers',type=int,default=1)
    parser.add_argument('--decoder_layers',type=int,default=1)
    parser.add_argument('--dropout_rate',type=float,default=0.2)
    parser.add_argument('--bidirectional',type=int,choices=[0,1],default=0)
    parser.add_argument('--encoder_name',type=str,default='lstm')
    parser.add_argument('--decoder_name',type=str,default='lstm')
    parser.add_argument('--preprocess_encoder_states',type=int,choices=[0,1],default=0)
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--decay',type=float,default=0.9)
    parser.add_argument('--checkpoint_path',type=str,default='m3/checkpoint/best_checkpoint.pt')
    parser.add_argument('--warm_start',action='store_true')
    parser.add_argument('--eval_epoch_int',type=int,default=1)
    parser.add_argument('--scheduler',type=str,default='reduce')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
    trainer.predict()

#%%