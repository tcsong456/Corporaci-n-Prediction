import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

dtypes = {'store_nbr':np.int8,
          'unit_sales':np.float32,
          'cluster':np.int8,
          'class':np.int16
          }

stores = pd.read_csv('data/stores.csv', dtype=dtypes)
items = pd.read_csv('data/items.csv', dtype=dtypes)
train = pd.read_csv('data/train.csv', parse_dates=['date'], dtype=dtypes)
test = pd.read_csv('data/test.csv', parse_dates=['date'], dtype=dtypes)

is_test = test.groupby(['store_nbr','item_nbr']).apply(lambda x:pd.Series({'is_test':1})).reset_index()
is_discrete = train.groupby('item_nbr')['unit_sales'].apply(lambda x:np.all(x.values.astype(int)==x.values))
is_discrete = is_discrete.reset_index().rename(columns={'unit_sales':'is_discrete'})
start_date = train.groupby(['store_nbr','item_nbr'])['date'].min()
test_start = test['date'].min()
test['unit_sales'] = -1

df = pd.concat([train,test])
del train,test
df['onpromotion'] = df['onpromotion'].replace(np.nan,2).map(int).astype(np.int8)
df = df.merge(is_test,how='left',on=['store_nbr','item_nbr'])
df = df[df['is_test']==1].drop('is_test',axis=1)

date_range = pd.date_range(df['date'].min(),df['date'].max(),freq='D')
date_idx = range(len(date_range))
dt_to_idx = dict(map(reversed,enumerate(date_range)))
test_start_idx = dt_to_idx[test_start]
df['date'] = df['date'].map(dt_to_idx).astype(np.int16)
df['unit_sales'] = df['unit_sales'].astype(np.float32)
missing_dates = list(set(date_idx) - set(df['date']))

df = df.pivot_table(index=['store_nbr','item_nbr'],columns='date')
fill = np.zeros([df.shape[0],len(missing_dates)])
fill[:] = np.nan
missing_df = pd.DataFrame(fill,columns=missing_dates)

op = pd.concat([df['onpromotion'].reset_index(),missing_df],axis=1).fillna(2)
op = op[['store_nbr','item_nbr'] + list(date_idx)]
op = op[date_idx].values.astype(np.int8)

for i in range(op.shape[1]):
    nan_mask = op[:,i] == 2
    p = 0.2 * op[~nan_mask,i].mean()
    if np.isnan(p):
        p = 0
    op[nan_mask,i] = np.random.binomial(n=1,p=p,size=nan_mask.sum())

uid = pd.concat([df['id'].reset_index(),missing_df],axis=1).fillna(0)
uid = uid[['store_nbr','item_nbr'] + list(date_idx)]

df = pd.concat([df['unit_sales'].reset_index(),missing_df],axis=1).fillna(0)
df = df[['store_nbr','item_nbr'] + list(date_idx)]

process_dir = 'data/processed'
if not os.path.exists(process_dir):
    os.makedirs(process_dir)

np.save(f'{process_dir}/x_raw.npy',df[date_idx].values.astype(np.float32))
np.save(f'{process_dir}/onpromotion.npy',op)
np.save(f'{process_dir}/id.npy',uid[date_idx].astype(np.int32))
del op,uid

df = pd.concat([df[['item_nbr','store_nbr']],np.log1p(np.maximum(df[date_idx],0).astype(np.float32))],axis=1)
np.save(f'{process_dir}/x.npy',df[date_idx].values)

start_date = start_date.reset_index().rename(columns={'date':'start_date'})
df = df.merge(start_date,how='left',on=['store_nbr','item_nbr'])
df['start_date'] = df['start_date'].map(lambda x:dt_to_idx.get(x,test_start_idx))

df = df.merge(is_discrete,how='left',on=['item_nbr'])
df['is_discrete'] = df['is_discrete'].fillna(0).astype(int)

store_columns = [col for col in stores.columns if col!='store_nbr']
stores[store_columns] = stores[store_columns].apply(lambda x:LabelEncoder().fit_transform(x))
df = df.merge(stores,how='left',on='store_nbr')

item_columns = ['family','class']
items[item_columns] = items[item_columns].apply(lambda x:LabelEncoder().fit_transform(x))
df = df.merge(items,how='left',on='item_nbr')
df['item_nbr'] = LabelEncoder().fit_transform(df['item_nbr'])

features = [
    ('store_nbr', np.int8),
    ('item_nbr', np.int32),
    ('city', np.int8),
    ('state', np.int8),
    ('type', np.int8),
    ('cluster', np.int8),
    ('family', np.int8),
    ('class', np.int16),
    ('perishable', np.int8),
    ('is_discrete', np.int8),
    ('start_date', np.int16)
]
for feat,dtype in features:
    vals = df[feat].values.astype(dtype)
    np.save(f'{process_dir}/{feat}.npy',vals)

x = df[date_idx].values
x_lags = [1,7,14]
lag_data = np.zeros([x.shape[0],x.shape[1],len(x_lags)],dtype=np.float32)
for i,lag in enumerate(x_lags):
    lag_data[:,lag:,i] = x[:,:-lag]
np.save(f'{process_dir}/x_lags.npy',lag_data)
del lag_data

xy_lags = [16,21,28,35,90,180,365,365*2,365*3]
lag_data = np.zeros([x.shape[0],x.shape[1],len(xy_lags)],dtype=np.float16)
for i,lag in enumerate(xy_lags):
    lag_data[:,lag:,i] = x[:,:-lag]
np.save(f'{process_dir}/xy_lags.npy',lag_data)
del lag_data

groups = [
    ['store_nbr'],
    ['item_nbr'],
    ['family'],
    ['class'],
    ['city'],
    ['state'],
    ['type'],
    ['cluster'],
    ['item_nbr', 'city'],
    ['item_nbr', 'type'],
    ['item_nbr', 'cluster'],
    ['family', 'city'],
    ['family', 'type'],
    ['family', 'cluster'],
    ['store_nbr', 'family'],
    ['store_nbr', 'class']
]
df_idx = df[['store_nbr', 'item_nbr', 'family', 'class', 'city', 'state', 'type', 'cluster']]
aux_data = np.zeros([df.shape[0],len(date_idx),len(groups)],dtype=np.float16)
for i,group in enumerate(groups):
    group_stat = df.groupby(group).mean()[date_idx].reset_index()
    group_stat = df_idx.merge(group_stat,how='left',on=group)
    aux_data[:,:,i] = group_stat[date_idx].fillna(0).values
np.save(f'{process_dir}/group_stat.npy',aux_data)
del aux_data

#%%
