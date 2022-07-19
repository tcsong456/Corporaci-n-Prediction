import numba
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

@numba.jit(nopython=False)
def single_corr(row,lag):
    s1 = row[lag:]
    s2 = row[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    output = np.sum(ds1 * ds2) / divider if divider !=0 else 0
    return output

@numba.jit(nopython=False)
def batch_corr(data,lag,starts,ends,threshold,backoffset=0):
    n_rows = data.shape[0]
    corr = np.empty(n_rows,dtype=np.float32)
    for i in range(n_rows):
        row = data[i]
        start,end = starts[i],ends[i]
        ratio = (end - start) / lag
        if ratio > threshold:
            series = row[start:end]
            c_1 = single_corr(series,lag)
            c_2 = single_corr(series,lag - 1)
            c_3 = single_corr(series,lag + 1)
            corr[i] = 0.5 * c_1 + c_2 * 0.25 + c_3 * 0.25
        else:
            corr[i] = np.nan
    return corr

dtypes = {'store_nbr':np.int8,
          'unit_sales':np.float32,
          'cluster':np.int8,
          'class':np.int16
          }
train = pd.read_csv('data/train.csv',parse_dates=['date'],dtype=dtypes)
test = pd.read_csv('data/test.csv',parse_dates=['date'],dtype=dtypes)
items = pd.read_csv('data/items.csv',dtype=dtypes)
stores = pd.read_csv('data/stores.csv',dtype=dtypes)

is_test = test.groupby(['store_nbr','item_nbr']).apply(lambda x:pd.Series({'is_test':1}))
train = train.merge(is_test,how='left',on=['store_nbr','item_nbr'])
train = train[train['is_test']==1].drop('is_test',axis=1)

date_range = pd.date_range(train['date'].min(),test['date'].max())
date_idx = dict(map(reversed,enumerate(date_range)))
missing_dates = set(date_range) - set(train['date']) - set(test['date'])
missing_columns = [date_idx.get(missing_date) for missing_date in missing_dates]
train['date'] = train['date'].map(date_idx)
train['unit_sales'] = np.log1p(np.maximum(train['unit_sales'],0))

test_length = (test['date'].max() - test['date'].min()).days + 1
test['date'] = test['date'].map(date_idx)
df = train.pivot_table(columns='date',values='unit_sales',index=['store_nbr','item_nbr']).fillna(0).astype(np.float32)
missing_data = pd.DataFrame(np.zeros([df.shape[0],len(missing_dates)]),columns=missing_columns,index=df.index)
df = pd.concat([df,missing_data],axis=1)
df_extra = pd.DataFrame(np.zeros([df.shape[0],test_length]),columns=list(np.arange(df.shape[1],len(date_range))),index=df.index)
df = pd.concat([df,df_extra],axis=1)
train['onpromotion'] = train['onpromotion'].fillna(0).astype(np.int8)

df_onpromo = train.pivot_table(columns='date',values='onpromotion',index=['store_nbr','item_nbr']).fillna(0).astype(np.int8)
df_onpromo = pd.concat([df_onpromo,missing_data],axis=1)
df_onpromo_test = test.pivot_table(columns='date',values='onpromotion',index=['store_nbr','item_nbr']).fillna(0).astype(np.int8)
df_onpromo = df_onpromo.merge(df_onpromo_test,how='left',on=['store_nbr','item_nbr'])

starts = train.groupby(['store_nbr','item_nbr'])['date'].min()
ends = train.groupby(['store_nbr','item_nbr'])['date'].max()
ref_si = pd.DataFrame(df.reset_index()[['store_nbr','item_nbr']])
starts = ref_si.merge(starts,how='left',on=['store_nbr','item_nbr']).set_index(['store_nbr','item_nbr']).values.flatten()
ends = ref_si.merge(ends,how='left',on=['store_nbr','item_nbr']).set_index(['store_nbr','item_nbr']).values.flatten()

df_id = test.pivot_table(index=['store_nbr','item_nbr'],columns='date',values='id')
df_id = ref_si.merge(df_id,how='left',on=['store_nbr','item_nbr']).set_index(['store_nbr','item_nbr'])

items_d = ref_si.merge(items,how='left',on='item_nbr').drop(['store_nbr'],axis=1)
data_dict = {item_col:LabelEncoder().fit_transform(items_d[item_col]).astype(np.int16) for item_col in items_d.columns}
stores_d = ref_si.merge(stores,how='left',on='store_nbr').drop(['item_nbr'],axis=1)
stores_dict = {store_col:LabelEncoder().fit_transform(stores_d[store_col]).astype(np.int16) for store_col in stores_d.columns}
data_dict.update(stores_dict)

max_date_idx = max(date_idx.values()) + 1
df = df[np.arange(max_date_idx)]
df_onpromo = df_onpromo[np.arange(max_date_idx)]

date_df = pd.DataFrame(date_range,columns=['date'])
dow = date_df['date'].dt.dayofweek
weekend = dow.map(lambda x:0 if x < 5 else 1)

aux_dict = {'starts':starts,
            'ends':ends,
            'data':df.values,
            'data_onpromo':df_onpromo.values,
            'dow':dow.values.astype(np.int8),
            'weekend':weekend.values.astype(np.int8),
            'id':df_id.values}
data_dict.update(aux_dict)

base_index = pd.DataFrame(np.arange(0,df.shape[1]),index=date_range,columns=['index'])
base_index = base_index.reset_index().rename(columns={'level_0':'date'})
def lag_index(lag):
    dates = date_range - pd.DateOffset(days=lag)
    cur_date = pd.DataFrame(dates,columns=['date'])
    cur_data = cur_date.merge(base_index,how='left',on='date').set_index('date')
    return cur_data.values
x_lags = [7,14]
x_lags_index = np.concatenate([lag_index(lag) for lag in x_lags],axis=1)
xy_lags = [16,30,60,90,180,360]
xy_lags_index = np.concatenate([lag_index(lag) for lag in xy_lags],axis=1)
lags = [7,14,30,60,90,180,360]
lag_index = np.concatenate([lag_index(lag) for lag in lags],axis=1)

year_autocorr = batch_corr(df.values,365,starts,ends,1.5)
quarter_autocorr = batch_corr(df.values,90,starts,ends,2)

temp_dict = {'lag_index':lag_index,
             'x_lag_index':x_lags_index,
             'xy_lag_index':xy_lags_index,
             'year_autocorr':year_autocorr,
             'quarter_autocorr':quarter_autocorr}
data_dict.update(temp_dict)

with open('m3/data_dict.pkl','wb') as f:
    pickle.dump(data_dict,f)
#%%
