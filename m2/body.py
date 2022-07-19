import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from m2.utils import logger
from m2.build_features import OnpromoMovingStats

class ValidateSubmit(OnpromoMovingStats):
    def __init__(self,
                 df,
                 df_onpromo,
                 mode='eval',
                 num_round=200,
                 num_decode_steps=16):
        super().__init__(data=df,
                         data_onpromo=df_onpromo,
                         periods=[4,10,20])
        
        assert mode in ['eval','submit']
        self.num_decode_steps = num_decode_steps
        self.df = df
        self.mode = mode
        
        max_date = max([col if isinstance(col,pd.Timestamp) else pd.Timestamp('2010-1-1') for col in df.columns])
        self.threshold_train_date = max_date - timedelta(days=2*num_decode_steps) if mode=='eval' else max_date - timedelta(days=num_decode_steps)
        self.threshold_val_date = max_date - timedelta(days=num_decode_steps)
        self.threshold_test_date = max_date
    
    def get_body_data(self,end_date,return_target=True):
        features = []
        for lag in [7,14,30,60,90,180,360]:
            lag_ = lag - 1
            train_start_date = end_date - timedelta(days=lag_)
            feature = self.build(train_start_date,end_date,prefix=f'lag_{lag}_')
            features.append(feature)
        features = pd.concat(features,axis=1).astype(np.float32)
        
        future_onpromo_means = []
        future_start_date = end_date + timedelta(days=1)
        for future_lag in [3,7,16]:
            furture_end_date = end_date + timedelta(days=future_lag)
            future_date_range = pd.date_range(future_start_date,furture_end_date)
            future_onpromo_data = self.data_onpromo[future_date_range]
            future_onpromo_mean = future_onpromo_data.mean(axis=1).to_frame('future_onpromo_mean').add_prefix(f'lag+{future_lag}')
            future_onpromo_means.append(future_onpromo_mean)
        future_onpromo_means = pd.concat(future_onpromo_means,axis=1)
        
        weekly_stats = self.weekly_build(end_date)
        features = pd.concat([features,weekly_stats,future_onpromo_means],axis=1)
        
        if return_target:
            target = self._get_target_data(end_date)
            return features,target
        return features
    
    def _get_target_data(self,start_date):
        target_start_date = start_date + timedelta(days=1)
        target_end_date = start_date + timedelta(days=self.num_decode_steps)
        target_dates_range = pd.date_range(target_start_date,target_end_date)
        target = self.df[target_dates_range].fillna(0).astype(np.float32).values
        return target
    
    @staticmethod
    def fit_validation(model,X_tr,X_val,y_tr,y_val):
        preds_val = []
        for i in range(y_tr.shape[1]):
            train_target = y_tr[:,i]
            model.fit(X_tr,train_target)
            pred_val = model.predict(X_val)
            preds_val.append(pred_val)
        
        preds_val = np.stack(preds_val,axis=1)
        error = mean_squared_error(y_val,preds_val)**0.5
        logger.info(f'validation loss:{error:.5f}')
    
    @staticmethod
    def fit_submission(model,X_tr,X_te,y_tr,df):
        test = pd.read_csv('data/test.csv',parse_dates=['date'])
        preds = []
        init_date = pd.Timestamp('2017-8-16')
        for i in range(y_tr.shape[1]):
            cur_date = init_date + timedelta(days=i)
            train_target = y_tr[:,i]
            model.fit(X_tr,train_target)
            pred = model.predict(X_te)
            pred = pd.DataFrame(pred,columns=['unit_sales'])
            pred['unit_sales'] = np.maximum(np.expm1(pred['unit_sales']),0)
            pred['date'] = cur_date
            pred.index=  df.index
            preds.append(pred)
        preds = pd.concat(preds).reset_index()
        preds = test.merge(preds,how='left',on=['store_nbr','item_nbr','date']).fillna(0)[['id','unit_sales']]
        preds.to_csv('submission.csv',index=False)
        return preds
        


#%%