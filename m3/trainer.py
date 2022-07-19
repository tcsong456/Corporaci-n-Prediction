import sys
import pickle
import os
import torch
import logging
import logzero
sys.path.append('.')
import numpy as np
import pandas as pd
from tqdm import tqdm
from m3.dataloader import CorporaciDS
from m3.model import TemporalSeqToSeq
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam,AdamW,SGD
from torch.optim.lr_scheduler import ExponentialLR,OneCycleLR,ReduceLROnPlateau

def create_logger():
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logger = logzero.setup_logger(formatter=formatter,
                                  level=logging.DEBUG,
                                  name='corporci')
    return logger
logger = create_logger()

class RMSE(nn.Module):
    def __init__(self,
                 predict_window):
        super().__init__()
        self.predict_window = predict_window
    
    def forward(self,y_true,y_pred,weight):
        weight = torch.repeat_interleave(weight,repeats=self.predict_window,dim=1)
        error = (y_true - y_pred)**2
        error *= weight
        loss = torch.sqrt(error.sum() / weight.sum())
        return loss

class EMA:
    def __init__(self,k=0.99):
        self.k = k
        self.state = None
        self.steps = 0
    
    def __call__(self,value):
        self.steps += 1
        if self.state is None:
            self.state = value
        else:
            eff_k = min(1 - 1 / self.steps,self.k)
            self.state = eff_k * self.state + (1 - eff_k) * value
        return self.state

class Metric:
    def __init__(self,decay=None):
        self.smoother = EMA(k=decay) if decay is not None else None
        self.last_epoch = 0
        self.epoch_values = []
        
    @property
    def avg_loss(self):
        return np.mean(self.epoch_values)
    
    def update(self,value,epoch):
        if self.smoother is not None:
            value = self.smoother(value)
        if epoch > self.last_epoch:
            self.epoch_values = []
            self.last_epoch = epoch
        self.epoch_values.append(value)

class Trainer:
    def __init__(self,
                 args):
        with open('m3/data_dict.pkl','rb') as f:
            data_dict = pickle.load(f)
            
        ds_params =  {'data_dict':data_dict,
                      'train_window':args.train_window,
                      'predict_window':args.predict_window,
                      'backoffset':args.backoffset}
        
        ds_train = CorporaciDS(**ds_params,
                               mode='train')
        self.dl_train = DataLoader(ds_train,batch_size=args.batch_size,shuffle=True)
        self.metric_train = Metric(decay=args.decay)
        self.train_len = len(ds_train)
        
        ds_eval = CorporaciDS(**ds_params,
                              mode='eval')
        self.dl_eval = DataLoader(ds_eval,batch_size=args.batch_size,shuffle=False)
        self.metric_eval = Metric(decay=args.decay)
        self.eval_len = len(ds_eval)
        
        ds_test = CorporaciDS(**ds_params,
                              mode='test')
        self.dl_test = DataLoader(ds_test,batch_size=args.batch_size,shuffle=False)
        self.test_len = len(self.dl_test)
        
        self.model = TemporalSeqToSeq(dl=self.dl_train,
                                      train_window=args.train_window,
                                      predict_window=args.predict_window,
                                      backoffset=args.backoffset,
                                      hidden_size=args.hidden_size,
                                      batch_size=args.batch_size,
                                      encoder_layers=args.encoder_layers,
                                      decoder_layers=args.decoder_layers,
                                      dropout_rate=args.dropout_rate,
                                      bidirectional=args.bidirectional,
                                      encoder_name=args.encoder_name,
                                      decoder_name=args.decoder_name,
                                      preprocess_encoder_states=args.preprocess_encoder_states).cuda()
        
        if args.optimizer.lower() == 'adam':
            optimizer = Adam(filter(lambda x:x.requires_grad,self.model.parameters()),lr=args.lr)
        elif args.optimizer.lower() == 'adamw':
            optimizer = AdamW(filter(lambda x:x.requires_grad,self.model.parameters()),lr=args.lr)
        elif args.optimizer.lower() == 'sgd':
            optimizer = SGD(filter(lambda x:x.requires_grad,self.model.parameters()),lr=args.lr)
        else:
            raise Exception('{args.optimizer} is not supported,pick on in [adam,adamw,sgd]')
        self.optimizer = optimizer
        
        if args.scheduler == 'exponential':
            scheduler = ExponentialLR(self.optimizer,gamma=0.5)
        elif args.scheduler == 'onecycle':
            scheduler = OneCycleLR(self.optimizer,max_lr=0.05,steps_per_epoch=len(self.dl_train),
                                   epochs=args.epochs)
        elif args.scheduler == 'reduce':
            scheduler = ReduceLROnPlateau(self.optimizer,mode='min',factor=0.2,patience=2)
        else:
            scheduler = None
        self.scheduler = scheduler
        
        self.loss_func = RMSE(predict_window=args.predict_window)
        self.best_loss = np.inf
        self.patience = 3
        self.bad_epoch = 0
        self.args = args
        self.start_epoch = 0
        
        if args.warm_start:
            if not os.path.exists('m3/checkpoint/best_checkpoint.pt'):
                raise Exception('save a best model checkpoint first')
            state_dict = torch.load(args.checkpoint_path)
            self.start_epoch = state_dict['best_epoch']
            self.best_loss = state_dict['best_loss']
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
        
    def _single_epoch_train(self,train_tq,epoch):
        self.model.train()
        for step,batch  in enumerate(train_tq):
            for k,v in batch.items():
                batch[k] = v.cuda()
                
            preds = self.model(batch)
            weight = batch['weight']
            if weight.dim() == 1:
                weight = weight[:,None]
            
            y_true = batch['y']
            loss = self.loss_func(y_true,preds,weight)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.metric_train.update(loss.item(),epoch)
            avg_loss = self.metric_train.avg_loss
            train_tq.set_postfix(loss=f'{avg_loss:.5f}')
            
            if self.args.scheduler == 'onecycle':
                self.scheduler.step()
        return avg_loss
    
    @torch.no_grad()
    def _single_epoch_eval(self,tq,epoch):
        self.model.eval()
        for step,batch in enumerate(tq):
            for k,v in batch.items():
                batch[k] = v.cuda()
            
            preds = self.model(batch)
            weight = batch['weight']
            if weight.dim() == 1:
                weight = weight[:,None]
            
            y = batch['y']
            loss = self.loss_func(y,preds,weight)
            self.metric_eval.update(loss.item(),epoch)
            avg_loss = self.metric_eval.avg_loss
            tq.set_postfix(loss=f'{avg_loss:.5f}')
        return avg_loss
    
    @torch.no_grad()
    def build_preds(self,tq):
        self.model.eval()
        preds,sid = [],[]
        for batch in tq:
            for k,v in batch.items():
                if k != 'id':
                    batch[k] = v.cuda()
            
            pred = self.model(batch)
            preds.append(pred)
            sid.append(batch['id'].detach().numpy())
        preds = torch.cat(preds)
        id = np.concatenate(sid)
        preds = preds.cpu().detach().numpy()
        preds = np.expm1(preds.flatten())
        id = id.flatten()
        content = np.stack([preds,id],axis=1)
        preds = pd.DataFrame(content,columns=['unit_sales','id'])
        return preds
    
    def _single_run(self,dataloader,data_len,single_epoch_func,epoch):
        inp_tq = tqdm(dataloader,total=data_len//self.args.batch_size+1)
        loss = single_epoch_func(inp_tq,epoch)
        inp_tq.close()
        return loss
    
    def end_epoch(self,eval_loss,epoch):
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            os.makedirs('m3/checkpoint',exist_ok=True)
            torch.save({'model':self.model.state_dict(),
                        'best_epoch':epoch,
                        'best_loss':self.best_loss,
                        'optimizer':self.optimizer.state_dict()},self.args.checkpoint_path)
            self.bad_epoch = 0
        else:
            self.bad_epoch += 1
            if self.bad_epoch > self.patience:
                raise Exception(f'{self.bad_epoch} number of bad epochs streak,early stopping triggered')
    
    def train(self):
        for epoch in range(self.start_epoch,self.args.epochs):
            logger.info(f'training epoch:{epoch}')
            train_loss = self._single_run(self.dl_train,self.train_len,self._single_epoch_train,epoch)
            logger.info(f'epoch:{epoch} train_loss:{train_loss:.5f}')
            
            if epoch % self.args.eval_epoch_int == 0:
                eval_loss = self._single_run(self.dl_eval,self.eval_len,self._single_epoch_eval,epoch)
                logger.info(f'epoch:{epoch} eval_loss:{eval_loss:.5f}')
            
                self.end_epoch(eval_loss,epoch)
                if self.args.scheduler == 'reduce':
                    self.scheduler.step(eval_loss)
                    lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f'current lr:{lr:.5f} reduce lr scheduler')
            
            if self.scheduler == 'exponential':
                self.scheduler.step()
    
    def predict(self):
        logger.info('predicting results for submission')
        state_dict = torch.load(self.args.checkpoint_path)
        self.model.load_state_dict(state_dict['model'])
        test_tq = tqdm(self.dl_test,total=self.test_len//self.args.batch_size+1)
        preds = self.build_preds(test_tq)
        
        ss = pd.read_csv('data/sample_submission.csv')[['id']]
        preds = ss.merge(preds,how='left',on='id').fillna(0)
        preds['unit_sales'] = np.clip(preds['unit_sales'],0,1000)
        preds.to_csv('m3/submission.csv',index=False)



#%%
