import os
import logging
import logzero
import pandas as pd
import numpy as np

LGB_PARAMS = {
            'num_leaves':80,
            'min_data_in_leaf':200,
            'feature_fraction':0.8,
            'bagging_fraction':0.7,
            'bagging_freq':1,
            'learning_rate':0.02,
            'metirc':'l2_root',
            'objective':'regression'
                }

def custome_logger(name):
    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logger = logzero.setup_logger(formatter=formatter,
                                  level=logging.DEBUG,
                                  name=name)
    return logger
logger = custome_logger('corporaci')

def load_data(root_path):
    dtypes = {'store_nbr':np.int8,
              'unit_sales':np.float32,
              'cluster':np.int8,
              'class':np.int16
              }
    
    outputs = {}
    for col in ['stores','items','train','test']:
        data_suffix = col + '.csv'
        data_path = os.path.join('data',data_suffix)
        if col == 'train' or col == 'test':
            data = pd.read_csv(data_path,dtype=dtypes,parse_dates=['date'])
        else:
            data = pd.read_csv(data_path,dtype=dtypes)
        outputs[col] = data
    
    test = outputs['test']
    test['onpromotion'] = test['onpromotion'].map(int).astype(np.int8)
    train = outputs['train']
    items,stores = outputs['items'],outputs['stores']
    
    train = train.merge(items,how='left',on='item_nbr').merge(stores,how='left',on='store_nbr')
    test = test.merge(items,how='left',on='item_nbr').merge(stores,how='left',on='store_nbr')
    outputs['train'] = train
    outputs['test'] = test
    
    return outputs

def build_dataset(group_features,agg_func,data_dict):
    features = '_'.join(group_features)
    logger.info(f'building {features} dataset')
    
    def pivot_data(data,value):
        df = data.pivot_table(columns='date',values=value,index=group_features,aggfunc=agg_func).astype(np.float32)
        missing_dates = set(pd.date_range(data['date'].min(),data['date'].max())) - set(df.columns.tolist())
        missing_data = pd.DataFrame(np.zeros([df.shape[0],len(missing_dates)],dtype=np.float32),columns=list(missing_dates),
                                     index=df.index)
        df = pd.concat([missing_data,df],axis=1)
        return df

    train,test = data_dict['train'],data_dict['test']    

    test_size = test.groupby(group_features).size().reset_index()
    test_columns = test_size.columns
    test_identifier = test_size.drop(test_columns[-1],axis=1)
    test_identifier['is_test'] = 1
    
    train = train.merge(test_identifier,how='left',on=group_features)
    train = train[train['is_test']==1].drop('is_test',axis=1)
    train = train.fillna(0)
    train['unit_sales'] = np.log1p(np.maximum(train['unit_sales'],0))

    df = pivot_data(train,'unit_sales')
    df_onpromo = pivot_data(train,'onpromotion').fillna(0)
    df_te_onpromo = pivot_data(test,'onpromotion')
    df_onpromo = df_onpromo.merge(df_te_onpromo,how='left',on=group_features)
    
    return df,df_onpromo


#%%
