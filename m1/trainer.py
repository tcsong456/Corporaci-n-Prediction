import torch
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam,SGD
from m1.tcn import TCN
from m1.utils import (RMSE,
                      Metric,
                      logger)
from m1.data_pipe import DataReader,BuildInp        

class Trainer:
    def __init__(self,
                 args):
        reader = DataReader(data_dir=args.data_dir,
                            test_size=args.test_size,
                            random_state=9867)
        build_input = BuildInp(reader=reader,
                               num_decode_steps=args.num_decode_steps)
        self.model = TCN(reader=reader,
                         receiptive_field=args.receiptive_field,
                         build_inp=build_input,
                         batch_size=args.batch_size,
                         use_bilstm=args.use_bilstm,
                         residual_channel=args.residual_channel,
                         num_decode_steps=args.num_decode_steps,
                         filter_widths=[2 for _ in range(9)],
                         dilations=[2**i for i in range(9)]).cuda()
        if args.optim == 'adam':
            self.optimizer = Adam(filter(lambda x:x.requires_grad,self.model.parameters()),lr=args.lr)
        elif args.optim == 'sgd':
            self.optimizer = SGD(filter(lambda x:x.requires_grad,self.model.parameters()),lr=args.lr)
        else:
            raise Exception(f'{args.optimizer} is not supported,pick one oe [adam,sgd]')
        
        self.num_decode_steps = args.num_decode_steps
        self.reader = reader
        self.build_input = build_input
        self.args = args
        self.early_stopping_epochs = args.early_stopping_epochs
        
        self.best_loss = np.inf
        self.best_epoch = 0
        self.bad_epoch = 0
        self.start_epoch = 0
        
        if args.warm_start:
            state_dict=  torch.load(f'{args.checkpoint_path}/best_checkpoint.pt')
            self.start_epoch = state_dict['best_epoch']
            self.best_loss = state_dict['best_loss']
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
    
    def _single_epoch_train(self,x_enc,x_dec,x,batch,x_mean,epoch):
        pred = self.model(x_enc,x_dec,x,batch,x_mean)
        y_lens = batch['y_len']
        weight = batch['weight']
        y_true = batch['y']
        
        mask = torch.zeros([self.model.encode_shape[0],self.num_decode_steps]).cuda()
        for i,y_len in enumerate(y_lens):
            mask[i,:int(y_len)] = 1
        
        loss_func = RMSE(mask,weight)
        loss = loss_func(y_true,pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_metric.update(loss.item(),epoch)
        loss = self.train_metric.avg_loss
        return loss
    
    def _single_epoch_test(self,x_enc,x_dec,x,batch,x_mean,epoch):
        pred = self.model(x_enc,x_dec,x,batch,x_mean)
        y_lens=  batch['y_len']
        weight = batch['weight']
        y_true = batch['y']
        
        mask = torch.zeros([self.model.encode_shape[0],self.num_decode_steps]).cuda()
        for i,y_len in enumerate(y_lens):
            mask[i,:int(y_len)] = 1
        
        loss_func = RMSE(mask,weight)
        loss = loss_func(y_true,pred)
        self.eval_metric.update(loss.item(),epoch)
        loss = self.eval_metric.avg_loss
        return loss
    
    def _train(self,dataloader,data_len,single_epoch_func,epoch):
        inp_tq = tqdm(dataloader,total=data_len)
        for batch in inp_tq:
            x_centered = self.build_input(batch,self.args.use_bilstm)
            x_mean = self.build_input.x_mean
            inp_package = {'x_enc':self.build_input.encode_features,
                           'x_dec':self.build_input.decode_features,
                           'x':x_centered,
                           'batch':batch,
                           'x_mean':x_mean,
                           'epoch':epoch}
            loss = single_epoch_func(**inp_package)
            inp_tq.set_postfix(loss=f'{loss:.5f}')
        inp_tq.close()
    
    def train(self):
        logger.info('running training epochs')
        train_data_len = len(self.reader.train_df)//self.args.batch_size
        eval_data_len = len(self.reader.val_df)//self.args.batch_size
        
        for epoch in range(self.start_epoch,self.args.epochs):
            self.model.train()
            self.train_metric = Metric(decay=self.args.decay)
            train_dataloader = self.reader.train_batch_generator(self.args.batch_size)
            self._train(train_dataloader,train_data_len,self._single_epoch_train,epoch)
            avg_train_loss = self.train_metric.avg_loss
            logger.info(f'epoch:{epoch} trian loss:{avg_train_loss:.5f}')
            
            if epoch % self.args.eval_interval_epoch == 0:
                self.eval_metric = Metric(decay=self.args.decay)
                eval_dataloader = self.reader.val_batch_generator(self.args.batch_size)
                with torch.no_grad():
                    self.model.eval()
                    self._train(eval_dataloader,eval_data_len,self._single_epoch_test,epoch)
                    avg_eval_loss = self.eval_metric.avg_loss
                    logger.info(f'epoch:{epoch} eval loss:{avg_eval_loss:.5f}')
            
            self.end_epoch(avg_eval_loss,epoch)
        
    def end_epoch(self,loss,epoch):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.bad_epoch = 0
            
            os.makedirs(self.args.checkpoint_path,exist_ok=True)
            checkpoint_save_path = f'{self.args.checkpoint_path}/best_checkpoint.pt'
            torch.save({'model':self.model.state_dict(),
                        'best_epoch':self.best_epoch,
                        'best_loss':self.best_loss,
                        'optimizer':self.optimizer.state_dict()},
                        checkpoint_save_path)
        else:
            self.bad_epoch += 1
            if self.bad_epoch >= self.early_stopping_epochs:
                raise Exception(f'{self.early_stopping_epochs} number of unimproved epochs,early stopping triggered')
        
        



#%%
