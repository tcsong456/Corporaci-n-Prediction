import torch
import logging
import logzero
import numpy as np
from torch import nn

def custom_logger(name):
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logger = logzero.setup_logger(formatter=formatter,
                                  name=name,
                                  level=logging.DEBUG)
    return logger

logger = custom_logger('corporaci')

class RMSE(nn.Module):
    def __init__(self,
                 mask,
                 weight):
        super().__init__()
        self.mask = mask
        self.weight = weight
    
    def forward(self,y_true,y_pred):
        error = (y_true - y_pred)**2
        weight = self.weight[:,None] * self.mask
        error *= weight
        loss = torch.sqrt(error.sum() / weight.sum())
        return loss

class EMA:
    def __init__(self,
                 decay=0.9):
        self.decay = decay
        self.steps = 0
        self.state = None
    
    def __call__(self,value):
        self.steps += 1
        if not self.state:
            self.state = value
        else:
            eff_k = min(1 - 1 / self.steps,self.decay)
            self.state =  eff_k * self.state + (1 - eff_k) * value
        return self.state

class Metric:
    def __init__(self,
                 decay=None):
        self.smoother = EMA(decay) if decay else None            
        self.last_epoch = -1
        self.epoch_value = []
        
    @property
    def avg_loss(self):
        return np.mean(self.epoch_value)
    
    def update(self,value,epoch):
        if self.smoother:
            value = self.smoother(value)
        if epoch > self.last_epoch:
            self.last_epoch = epoch
            self.epoch_value = []
        self.epoch_value.append(value)
    
        
        

#%%