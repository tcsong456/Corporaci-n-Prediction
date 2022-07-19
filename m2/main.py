import os
import sys
sys.path.append('.')
import warnings 
warnings.filterwarnings(action='ignore')
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
from m2.utils import LGB_PARAMS,logger,load_data,build_dataset
from m2.model import Lightgbm
from m2.body import ValidateSubmit

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,choices=['eval','submit'],default='eval')
    parser.add_argument('--data_dir',type=str,default='data')
    parser.add_argument('--num_rounds',type=int,default=200)
    parser.add_argument('--backwards',type=int,default=6)
    parser.add_argument('--backwards_length',type=int,default=7)
    parser.add_argument('--num_decode_steps',type=int,default=16)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    data_dict = load_data(root_path=args.data_dir)
    df,df_onpromo = build_dataset(group_features=['store_nbr','item_nbr'],
                              agg_func='mean',
                              data_dict=data_dict)
    df_items,df_items_onpromo = build_dataset(group_features=['item_nbr'],
                                          agg_func='sum',
                                          data_dict=data_dict)
    df_stores_class,df_stores_class_onpromo = build_dataset(group_features=['store_nbr','class'],
                                                        agg_func='sum',
                                                        data_dict=data_dict)
    del data_dict
    
    items = pd.read_csv(os.path.join(args.data_dir,'items.csv'))
    store_item = df.reset_index()[['store_nbr','item_nbr']]
    store_item_class = store_item.merge(items,how='left',on=['item_nbr'])[['store_nbr','item_nbr','class']]
    
    df = df.reset_index().merge(items,how='left',on='item_nbr').set_index(['store_nbr','item_nbr'])
    weight = pd.concat([df['perishable']]*args.backwards) * 0.25 + 1
    model = Lightgbm(params=LGB_PARAMS,
                     num_round=args.num_rounds,
                     weight=weight)
    
    y_val = None
    runners = []
    for data,data_onpromo in zip([df,df_items,df_stores_class],[df_onpromo,df_items_onpromo,df_stores_class_onpromo]):
        runner = ValidateSubmit(df=data,
                                df_onpromo=data_onpromo,
                                mode=args.mode,
                                num_round=args.num_rounds,
                                num_decode_steps=args.num_decode_steps)
        runners.append(runner)
    
    X_tr,targets = [],[]
    gp_columns = [['store_nbr','item_nbr'],['item_nbr'],['store_nbr','class']]
    for i in range(args.backwards):
        logger.info(f'running backward data:{i}')
        data_tr = []
        for runner,gp in zip(runners,gp_columns):
            train_end_date = runner.threshold_train_date - timedelta(i*args.backwards_length)
            feature,target = runner.get_body_data(train_end_date,return_target=True)
            data_tr.append(store_item_class.merge(feature,how='left',on=gp).set_index(['store_nbr','item_nbr']).fillna(0).drop('class',axis=1))
            if gp == ['store_nbr','item_nbr']:
                targets.append(target)
        data_tr = pd.concat(data_tr,axis=1)
        X_tr.append(data_tr)
    X_tr = pd.concat(X_tr).astype(np.float32)
    y_tr = np.concatenate(targets).astype(np.float32)
    
    
    if args.mode == 'eval':
        X_val = []
        for runner,gp in zip(runners,gp_columns):
            if gp == ['store_nbr','item_nbr']:
                x_val,y_val = runner.get_body_data(runner.threshold_val_date,return_target=True)
            else:
                x_val = runner.get_body_data(runner.threshold_val_date,return_target=False)
            X_val.append(store_item_class.merge(x_val,how='left',on=gp).set_index(['store_nbr','item_nbr']).fillna(0).drop('class',axis=1))
        X_val = pd.concat(X_val,axis=1)
        runner.fit_validation(model,X_tr,X_val,y_tr,y_val)
    else:
        X_te = []
        for runner,gp in zip(runners,gp_columns):
            x_te = runner.get_body_data(runner.threshold_test_date,return_target=False)
            X_te.append(store_item_class.merge(x_te,how='left',on=gp).set_index(['store_nbr','item_nbr']).fillna(0).drop('class',axis=1))
        X_te = pd.concat(x_te,axis=1)
        runner.fit_submission(model,X_tr,X_te,y_tr,df)
    


#%%

