import os
import copy
import torch
import numpy as np
from m1.utils import logger
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

class DataFrame:
    def __init__(self,
                 data,
                 columns,
                 idx=None):
        assert len(columns) == len(data),'provided number of columns must equal to the number of data'
        lengths = [mat.shape[0] for mat in data]
        assert len(set(lengths))==1,'all sub data must have the same first dimension'
        
        self.data = data
        self.columns = columns
        self.dict = dict(zip(columns,data))
        self.length = lengths[0]
        self.idx = idx if idx is not None else np.arange(self.length)
    
    def shapes(self):
        return dict(zip(self.columns,[mat.shape for mat in self.data]))
    
    def dtypes(self):
        return dict(zip(self.columns,[mat.dtype for mat in self.data]))
    
    def train_test_split(self,test_size,random_state=1234):
        train_idx,test_idx = train_test_split(self.idx,test_size=test_size,random_state=random_state)
        train_df = DataFrame(self.data,copy.copy(self.columns),idx=train_idx)
        test_df = DataFrame(self.data,copy.copy(self.columns),idx=test_idx)
        return train_df,test_df
    
    def shuffle(self):
        np.random.shuffle(self.idx)
    
    def batch_generator(self,batch_size,allow_small_batch=False,shuffle=True):
        if shuffle:
            self.shuffle()
        for i in range(0,len(self.idx),batch_size):
            idx = self.idx[i:i+batch_size]
            if not allow_small_batch and len(idx) < batch_size:
                break
            yield DataFrame([mat[idx].copy() for mat in self.data],copy.copy(self.columns))
    
    def items(self):
        return self.dict.items()
    
    def __iter__(self):
        return self.dict.items().__iter__()
    
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self,key):
        assert isinstance(key,str),'input key must be string'
        return self.dict[key]
    
    def __setitem__(self,key,value):
        assert value.shape[0]==self.length,'input value must have the same first dimention with current data'
        if key not in self.columns:
            self.columns.append(key)
            self.data.append(value)
        self.dict[key] = value

class DataReader:
    def __init__(self,
                 data_dir,
                 test_size=0.25,
                 random_state=7519):
        data_cols = [
            'x_raw',
            'onpromotion',
            'id',
            'x',
            'store_nbr',
            'item_nbr',
            'city',
            'state',
            'type',
            'cluster',
            'family',
            'class',
            'perishable',
            'is_discrete',
            'start_date',
            'x_lags',
            'xy_lags',
            'group_stat',
        ]
        logger.info('loading data')
        data = [np.load(os.path.join(data_dir,f'{data_col}.npy')) for data_col in data_cols]
        
        self.test_df = DataFrame(data=data,columns=data_cols)
        self.train_df,self.val_df = self.test_df.train_test_split(test_size=test_size,random_state=random_state)
        
        self.num_city = self.test_df['city'].max() + 1
        self.num_state = self.test_df['state'].max() + 1
        self.num_type = self.test_df['type'].max() + 1
        self.num_cluster = self.test_df['cluster'].max() + 1
        self.num_family = self.test_df['family'].max() + 1
        self.num_item_class = self.test_df['class'].max() + 1
        self.num_perishable = self.test_df['perishable'].max() + 1
        self.num_store_nbr = self.test_df['store_nbr'].max() + 1
        self.num_item_nbr = self.test_df['item_nbr'].max() + 1
    
    def convert_to_torch(self,x,cuda=True,dtype=None):
        if dtype:
            x = x.astype(dtype)
        x = torch.from_numpy(x)
        if cuda:
            x = x.cuda()
        return x
        
    def train_batch_generator(self,batch_size):
        return self.generate_batch(batch_size=batch_size,
                                   df=self.train_df,
                                   mode='train',
                                   shuffle=True)
    
    def val_batch_generator(self,batch_size):
        return self.generate_batch(batch_size=batch_size,
                                   df=self.val_df,
                                   mode='val',
                                   shuffle=False)
    
    def test_batch_generator(self,batch_size):
        return self.generate_batch(batch_size=batch_size,
                                   df=self.test_df,
                                   mode='test',
                                   shuffle=False)
    
    def generate_batch(self,batch_size,df,mode,shuffle=True):
        batch_gen = df.batch_generator(batch_size=batch_size,
                                       allow_small_batch=(mode=='test'),
                                       shuffle=shuffle)
        for batch in batch_gen:
            num_decode_steps = 16
            max_encode_len = batch['x'].shape[1] - num_decode_steps
            
            x = np.zeros([len(batch),max_encode_len])
            y = np.zeros([len(batch),num_decode_steps])
            x_raw = np.zeros([len(batch),max_encode_len])
            x_lags = np.zeros([len(batch),max_encode_len,batch['x_lags'].shape[2]+batch['xy_lags'].shape[2]])
            y_lags = np.zeros([len(batch),num_decode_steps,batch['xy_lags'].shape[2]])
            x_op = np.zeros([len(batch),max_encode_len])
            y_op = np.zeros([len(batch),num_decode_steps])
            x_len = np.zeros([len(batch)])
            y_len = np.zeros([len(batch)])
            y_id = np.zeros([len(batch),num_decode_steps])
            x_ts = np.zeros([len(batch),max_encode_len,batch['group_stat'].shape[2]])
            weights = np.zeros([len(batch)])
            weights[batch['perishable']==1] = 1.25
            weights[batch['perishable']==0] = 1
            
            for i,(data,data_raw,start_idx,x_lag,xy_lag,op,uid,gs) in enumerate(zip(
                    batch['x'],batch['x_raw'],batch['start_date'],batch['x_lags'],
                    batch['xy_lags'],batch['onpromotion'],batch['id'],batch['group_stat'])):
                seq_len = max_encode_len - start_idx
                val_window = 365
                train_window = 365
                
                if mode == 'train':
                    if seq_len == 0:
                        rand_encode_len = 0
                        weights[i] = 0
                    elif seq_len <= train_window:
                        rand_encode_len = np.random.randint(0,seq_len)
                    else:
                        rand_encode_len = np.random.randint(seq_len-train_window,seq_len)
                    rand_decode_len = min(seq_len-rand_encode_len,num_decode_steps)
                elif mode == 'val':
                    if seq_len <= num_decode_steps:
                        rand_encode_len = 0
                        weights[i] = 0
                    elif seq_len <= val_window + num_decode_steps:
                        rand_encode_len = np.random.randint(0,seq_len-num_decode_steps)
                    else:
                        rand_encode_len = np.random.randint(seq_len-(val_window+num_decode_steps),seq_len-num_decode_steps)
                    rand_decode_len = min(seq_len-rand_encode_len,num_decode_steps)
                elif mode == 'test':
                    rand_encode_len = seq_len
                    rand_decode_len = num_decode_steps
                
                end_idx = start_idx + rand_encode_len
                x[i,:rand_encode_len] = data[start_idx:end_idx]
                y[i:,:rand_decode_len] = data[end_idx:end_idx+rand_decode_len]
                x_raw[i,:rand_encode_len] = data_raw[start_idx:end_idx]
                
                x_lags[i,:rand_encode_len,:x_lag.shape[1]] = x_lag[start_idx:end_idx,:]
                x_lags[i,:rand_encode_len,x_lag.shape[1]:] = xy_lag[start_idx:end_idx,:]
                y_lags[i,:rand_decode_len,:] = xy_lag[end_idx:end_idx+rand_decode_len,:]
                
                x_op[i,:rand_encode_len] = op[start_idx:end_idx]
                y_op[i,:rand_decode_len] = op[end_idx:end_idx+rand_decode_len]
                x_ts[i,:rand_encode_len,:] = gs[start_idx:end_idx,:]
                y_id[i,:rand_decode_len] = uid[end_idx:end_idx+rand_decode_len]
                x_len[i] = end_idx - start_idx
                y_len[i] = rand_decode_len
            
            batch['x'] = self.convert_to_torch(x,dtype=np.float32)
            batch['x_raw'] = self.convert_to_torch(x_raw,dtype=np.float32)
            batch['y'] = self.convert_to_torch(y,dtype=np.float32)
            batch['x_lags'] = self.convert_to_torch(x_lags,dtype=np.float32)
            batch['y_lags'] = self.convert_to_torch(y_lags,dtype=np.float32)
            batch['x_len'] = x_len
            batch['y_len'] = y_len
            batch['group_stat'] = self.convert_to_torch(x_ts,dtype=np.float32)
            batch['weight'] = self.convert_to_torch(weights)
            batch['x_op'] = self.convert_to_torch(x_op,dtype=np.int8)
            batch['y_op'] = self.convert_to_torch(y_op,dtype=np.int8)
            
            for item in ['city','state','type','cluster','family','class','perishable',
                         'is_discrete','store_nbr','item_nbr']:
                batch[item] = self.convert_to_torch(batch[item],dtype=np.int16)
                
            yield batch
            
class  BuildInp():
    def __init__(self,
                 reader,
                 num_decode_steps
                 ):
        self.num_decode_steps = num_decode_steps
        
        self.reader = reader
        self.item_class_embedding = nn.Embedding(reader.num_item_class,20).cuda()
        self.item_nbr_embedding = nn.Embedding(reader.num_item_nbr,50).cuda()
    
    def _process_one_hot(self,x,repeats=100,num_classes=0):
        if x.dtype != torch.long:
            x = x.long()
        oh_data = F.one_hot(x,num_classes=num_classes)[:,None,:]
        oh_data = torch.repeat_interleave(oh_data,repeats,dim=1)
        return oh_data
    
    def __call__(self,batch,use_bilstm=False):
        x = batch['x']
        x_mean = torch.Tensor([x[:int(x_len)].mean() if x_len > 0 else 0 for x,x_len in zip(batch['x'],batch['x_len'])])[:,None].cuda()
        x_centered = x - x_mean
        x_ts_centered = batch['group_stat'] - x_mean[:,None]
        x_lags_centered = batch['x_lags'] - x_mean[:,None]
        y_lags_centered = batch['y_lags'] - x_mean[:,None]
        x_is_zero = (batch['x_raw'] == 0) * 1
        x_is_negative = (batch['x_raw'] < 0) * 1
        x_oh_op = F.one_hot(batch['x_op'].long())
        y_oh_op = F.one_hot(batch['y_op'].long())
        
        item_class = self.item_class_embedding(batch['class'].long())
        item_nbr = self.item_nbr_embedding(batch['item_nbr'].long())
        self.encode_features = torch.cat([x_ts_centered,x_lags_centered,x_oh_op,x_is_zero[:,:,None],
                                          x_is_negative[:,:,None],x_centered[:,:,None],
                                          torch.repeat_interleave(x_mean[:,None,:],x.shape[1],dim=1),
                                          self._process_one_hot(batch['city'],x.shape[1],self.reader.num_city),
                                          self._process_one_hot(batch['state'],x.shape[1],self.reader.num_state),
                                          self._process_one_hot(batch['type'],x.shape[1],self.reader.num_type),
                                          self._process_one_hot(batch['cluster'],x.shape[1],self.reader.num_cluster),
                                          self._process_one_hot(batch['family'],x.shape[1],self.reader.num_family),
                                          self._process_one_hot(batch['perishable'],x.shape[1],2),
                                          self._process_one_hot(batch['is_discrete'],x.shape[1],2),
                                          self._process_one_hot(batch['store_nbr'],x.shape[1],self.reader.num_store_nbr),
                                          torch.repeat_interleave(item_class[:,None,:],x.shape[1],dim=1),
                                          torch.repeat_interleave(item_nbr[:,None,:],x.shape[1],dim=1)],axis=2)

        self.decode_features = torch.cat([y_lags_centered,y_oh_op,
                                          torch.repeat_interleave(x_mean[:,None,:],self.num_decode_steps,dim=1),
                                          self._process_one_hot(batch['city'],self.num_decode_steps,self.reader.num_city),
                                          self._process_one_hot(batch['state'],self.num_decode_steps,self.reader.num_state),
                                          self._process_one_hot(batch['type'],self.num_decode_steps,self.reader.num_type),
                                          self._process_one_hot(batch['cluster'],self.num_decode_steps,self.reader.num_cluster),
                                          self._process_one_hot(batch['family'],self.num_decode_steps,self.reader.num_family),
                                          self._process_one_hot(batch['perishable'],self.num_decode_steps,2),
                                          self._process_one_hot(batch['is_discrete'],self.num_decode_steps,2),
                                          self._process_one_hot(batch['store_nbr'],self.num_decode_steps,self.reader.num_store_nbr),
                                          torch.repeat_interleave(item_class[:,None,:],self.num_decode_steps,dim=1),
                                          torch.repeat_interleave(item_nbr[:,None,:],self.num_decode_steps,dim=1)],axis=2)

        self.x_mean = x_mean
        if use_bilstm:
            bi_lstm = nn.LSTM(input_size=self.decode_features.shape[-1],
                              hidden_size=50).cuda()
            bi_lstm_features,*_ = bi_lstm(self.decode_features)
            self.decode_features = torch.cat([self.decode_features,bi_lstm_features],axis=2)
        return x_centered

#%%
