import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder

class CorporaciDS(Dataset):
    def __init__(self,
                 data_dict,
                 train_window,
                 predict_window,
                 backoffset=0,
                 mode='train'):
        assert mode in ['train','eval','test'],'{mode} is not supported,pick one from [train,eval,test]'
        self.data_dict = data_dict
        self.train_window = train_window
        self.predict_window = predict_window
        self.backoffset = backoffset
        self.mode = mode
        
        self.family_num = self.data_dict['family'].max() + 1
        self.store_num = self.data_dict['store_nbr'].max() + 1
        self.city_num = self.data_dict['city'].max() + 1
        self.state_num = self.data_dict['state'].max() + 1
        self.type_num = self.data_dict['type'].max() + 1
        self.cluster_num = self.data_dict['cluster'].max() + 1
        self.dow_num = self.data_dict['dow'].max() + 2
    
    def _build_lag_data(self,lag_index,start,end,ind):
        li = lag_index[:,ind][start:end]
        nan_lag_mask = np.isnan(li)
        li = np.where(nan_lag_mask,np.zeros_like(li),li).astype(int)
        lag_data = self.data[li]
        lag_data[nan_lag_mask] = 0
        return lag_data
    
    def _build_dow_oh_data(self,start,end,add_missing_data=False,data_len=None):
        dow = self.data_dict['dow'][start:end]
        if add_missing_data:
            if len(dow) < self.train_window:
                missing_dow_len = self.train_window - len(dow)
                missing_dow = [-1] * missing_dow_len
                dow = np.array(dow.tolist() + missing_dow,dtype=np.int8)
        dow_oh = np.zeros([data_len,self.dow_num],dtype=np.int8)
        for i,j in zip(range(len(dow)),dow):
            dow_oh[i][j] = 1
        return dow_oh
    
    def _build_op_data(self,start,end,add_missing_data=False):
        op = self.data_onpromo[start:end]
        if add_missing_data:
            if len(op) < self.train_window:
                missing_op_len = self.train_window - len(op)
                missing_op = [-1] * missing_op_len
                op = np.array(op.tolist() + missing_op,dtype=np.int8)
        return op
    
    def __getitem__(self,index):
        sid = self.data_dict['id'][index]
        start = self.data_dict['starts'][index]
        end = self.data_dict['ends'][index]
        data = self.data_dict['data'][index]
        data_onpromo = self.data_dict['data_onpromo'][index]
        perishable = self.data_dict['perishable'][index]
        lag_index = self.data_dict['lag_index']
        self.data = data
        self.data_onpromo = data_onpromo
        
        family = self.data_dict['family'][index]
        family_oh = np.zeros([self.family_num],dtype=np.int8)
        family_oh[family] = 1
        
        store = self.data_dict['store_nbr'][index]
        store_oh = np.zeros([self.store_num],dtype=np.int8)
        store_oh[store] = 1
        
        city = self.data_dict['city'][index]
        city_oh = np.zeros([self.city_num],dtype=np.int8)
        city_oh[city] = 1
        
        state = self.data_dict['state'][index]
        state_oh = np.zeros([self.state_num],dtype=np.int8)
        state_oh[state] = 1
        
        tp = self.data_dict['type'][index]
        type_oh = np.zeros([self.type_num],dtype=np.int8)
        type_oh[tp] = 1
        
        cluster = self.data_dict['cluster'][index]
        cluster_oh = np.zeros([self.cluster_num],dtype=np.int8)
        cluster_oh[cluster] = 1
        
        year_corr = self.data_dict['year_autocorr'][index]
        quarter_corr = self.data_dict['quarter_autocorr'][index]
        corr = np.array([year_corr,quarter_corr])

        weight = None
        x = np.zeros([self.train_window],dtype=np.float32)
        if self.mode == 'train':
            available_days = end - start - self.backoffset - 2 * self.predict_window
        elif self.mode == 'eval':
#            full_length = self.data_dict['data'].shape[1] - self.predict_window
            start = end - self.train_window - self.backoffset - self.predict_window
            available_days = self.train_window if start > 0 else 0
#            start = full_length - self.train_window - self.predict_window - self.backoffset
        else:
#            full_length = self.data_dict['data'].shape[1] - self.predict_window
            start = end - self.train_window
            available_days = self.train_window if start > 0 else 0
        
        oh_encoder = OneHotEncoder()
        self.oh_encoder = oh_encoder
        encode_len = 0
        if available_days <= 0:
            weight = 0
            encode_len = 0
            end_offset = start + self.train_window
            x_dow_oh = np.zeros([self.train_window,self.dow_num],dtype=np.int8)
            y_dow_oh = np.zeros([self.predict_window,self.dow_num],dtype=np.int8)
            x_lag_data = np.zeros([self.train_window,lag_index.shape[1]],dtype=np.float32)
            y_lag_data = np.zeros([self.predict_window,lag_index.shape[1]],dtype=np.float32)
            x_op = np.zeros([self.train_window],dtype=np.int8)
            y_op = np.zeros([self.predict_window],dtype=np.int8)
            end_offset = 0
        elif available_days <= self.train_window:
            encode_len = available_days
            end_offset = start + encode_len
            x[:encode_len] = data[start:end_offset]
            
            x_op = self._build_op_data(start,end_offset,add_missing_data=True)
            y_op = self._build_op_data(end_offset,end_offset+self.predict_window)
            x_dow_oh = self._build_dow_oh_data(start,end_offset,add_missing_data=True,data_len=self.train_window)
            y_dow_oh = self._build_dow_oh_data(end_offset,end_offset+self.predict_window,data_len=self.predict_window)
            
            x_lag_datas,y_lag_datas = [],[]
            for i in range(lag_index.shape[1]):
                x_lag_data = self._build_lag_data(lag_index,start,end_offset,i)
                y_lag_data = self._build_lag_data(lag_index,end_offset,end_offset+self.predict_window,i)
                if x_lag_data.shape[0] < self.train_window:
                    missing_lag_len = self.train_window - x_lag_data.shape[0]
                    missing_lag = np.zeros([missing_lag_len])
                    x_lag_data = np.concatenate([x_lag_data,missing_lag])
                x_lag_datas.append(x_lag_data),y_lag_datas.append(y_lag_data)
            x_lag_data = np.stack(x_lag_datas,axis=1)
            y_lag_data = np.stack(y_lag_datas,axis=1)
        else:
            encode_len = self.train_window
            start_offset = np.random.randint(0,available_days-self.train_window)
            start_offset = start + start_offset
            end_offset = start_offset + self.train_window
            x = data[start_offset:end_offset]
            
            x_op = self._build_op_data(start_offset,end_offset)
            y_op = self._build_op_data(end_offset,end_offset+self.predict_window)
            x_dow_oh = self._build_dow_oh_data(start_offset,end_offset,
                                               data_len=self.train_window)
            y_dow_oh = self._build_dow_oh_data(end_offset,end_offset+self.predict_window,
                                               data_len=self.predict_window)
            
            x_lag_datas,y_lag_datas = [],[]
            for i in range(lag_index.shape[1]):
                x_lag_data = self._build_lag_data(lag_index,start_offset,end_offset,i)
                y_lag_data = self._build_lag_data(lag_index,end_offset,end_offset+self.predict_window,i)
                x_lag_datas.append(x_lag_data),y_lag_datas.append(y_lag_data)
            x_lag_data = np.stack(x_lag_datas,axis=1)
            y_lag_data = np.stack(y_lag_datas,axis=1)
        
        if self.mode != 'test':
            y = data[end_offset:end_offset+self.predict_window]
        else:
            y = np.zeros([self.predict_window])
            
        if weight is None:
            weight = 1.0 if perishable==0 else 1.25
        x_len = np.array(encode_len)
        weight = np.array(weight)
        mean = x.mean()
        std = np.sqrt(np.mean((x - mean)**2))
        norm_x = (x - mean) / std
        norm_x = np.where(np.isnan(norm_x)|np.isinf(norm_x),np.zeros_like(norm_x),norm_x)
        x_norm_lag = (x_lag_data - mean) / std
        y_norm_lag = (y_lag_data - mean) / std
        
        encode_features = np.concatenate([norm_x[:,None],x_norm_lag,x_op[:,None],np.tile(family_oh,(self.train_window,1)),
                                          np.tile(store_oh,(self.train_window,1)),np.tile(city_oh,(self.train_window,1)),
                                          np.tile(state_oh,(self.train_window,1)),np.tile(type_oh,(self.train_window,1)),
                                          np.tile(cluster_oh,(self.train_window,1)),x_dow_oh,np.tile(corr,(self.train_window,1))
                                          ],axis=1).astype(np.float32)
        decode_features = np.concatenate([y_norm_lag,y_op[:,None],np.tile(family_oh,(self.predict_window,1)),
                                          np.tile(store_oh,(self.predict_window,1)),np.tile(city_oh,(self.predict_window,1)),
                                          np.tile(state_oh,(self.predict_window,1)),np.tile(type_oh,(self.predict_window,1)),
                                          np.tile(cluster_oh,(self.predict_window,1)),y_dow_oh,np.tile(corr,(self.predict_window,1))],axis=1)
        encode_features = np.where(np.isnan(encode_features)|np.isinf(encode_features),np.zeros_like(encode_features),encode_features)
        decode_features = np.where(np.isnan(decode_features)|np.isinf(decode_features),np.zeros_like(decode_features),decode_features)
        outputs =  {'encode_features':encode_features,
                    'decode_features':decode_features,
                    'x_len':x_len,
                    'weight':weight,
                    'y':y,
                    'x':norm_x,
                    'mean':mean,
                    'std':std,
                    'id':sid}
        return outputs

    
    def __len__(self):
        return self.data_dict['data'].shape[0]

#%%
