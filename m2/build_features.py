import pandas as pd
import numpy as np
from datetime import timedelta
from m2.utils import logger
from numba import jit

class Base:
    def _get_decay_average(self,data,decay):
        decay_value = decay**np.arange(data.shape[1])[::-1]
        decay_data = data * decay_value
        mean = decay_data.sum(axis=1) / decay_value.sum()
        return mean
    
    @jit(nopython=False)
    def find_start_end(self,data):
        stats = np.zeros([data.shape[0],3],dtype=np.float32)
        for ind,row in enumerate(data):
            for i in range(len(row)):
                if row[i] > 0:
                    break
            for j in range(len(row)-1,-1,-1):
                if row[j] > 0:
                    break
            if j > i:
                valid_len = j - i + 1
            else:
                valid_len = 0
            valid_len_ratio = valid_len / data.shape[1]
            stats[ind] = np.array([j,valid_len,valid_len_ratio])
        stats = pd.DataFrame(stats,columns=['end','valid_len','valid_Len_ratio'])
        stats.index = self.data.index
        return stats
    
    def get_stats(self,data,decay=None):
        median = data.median(axis=1)
        min = data.min(axis=1)
        max = data.max(axis=1)
        std = data.std(axis=1)
        decay_mean = self._get_decay_average(data,decay)
        mean = data.mean(axis=1)
        sales_days = (data > 0).sum(axis=1)
        sales_days_ratio = sales_days / data.shape[1]
        start_end_info = self.find_start_end(data.values)
        
        columns = ['min','max','mean','decay_mean','median','std',
                   'sales_days','sales_days_ratio']
        stats = pd.concat([min,max,mean,decay_mean,median,std,sales_days,sales_days_ratio],axis=1)
        stats.columns = columns
        stats = pd.concat([stats,start_end_info],axis=1)
        return stats
    
    def get_weekly_stats(self,max_date,period):
        weekly_stats = []
        for i in range(7):
            cur_date = max_date - timedelta(days=i)
            weekly_dates = pd.date_range(cur_date,freq='-7D',periods=period)
            d = self.data[weekly_dates]
            weekly_decay_mean = self._get_decay_average(d,decay=0.992)
            mean = d.mean(axis=1).to_frame('mean')
            stat = pd.concat([weekly_decay_mean,mean],axis=1).add_prefix(f'week{i}_')
            weekly_stats.append(stat)
        weekly_stats = pd.concat(weekly_stats,axis=1)
        return weekly_stats
    
    def get_onpromo_data(self,data,data_onpromo,decay):
        onpromo_mean = data_onpromo.mean(axis=1).to_frame('onpromo_mean')
        onpromo_sum = data_onpromo.sum(axis=1).to_frame('onpromo_sum')
        
        has_onpromo_mean = (data_onpromo * data).sum(axis=1) / data_onpromo.sum(axis=1)
        no_onpromo_mask = data_onpromo == 0
        no_onpromo_mean = (no_onpromo_mask * data).sum(axis=1) / no_onpromo_mask.sum(axis=1)
        
        decay_value = decay**np.arange(data.shape[1])[::-1]
        decay_data = data * decay_value
        decay_onpromo = data_onpromo * decay_value
        has_onpromo_decay_mean = (decay_data * data_onpromo).sum(axis=1) / decay_onpromo.sum(axis=1)
        
        decay_no_onpromo = no_onpromo_mask * decay_value
        no_onpromo_decay_mean = (decay_data * no_onpromo_mask).sum(axis=1) / decay_no_onpromo.sum(axis=1)
        
        onpromo_start_end = self.find_start_end(data_onpromo.values).add_prefix('onpromo_')
        
        data = pd.concat([onpromo_mean,has_onpromo_mean,no_onpromo_mean,has_onpromo_decay_mean,
                          no_onpromo_decay_mean,onpromo_sum,onpromo_start_end],axis=1)
        return data
        

class TemporalMovingStats(Base):
    def __init__(self,
                 data,
                 ):
        self.data = data

    def build(self,min_date,max_date,prefix=None):
        date_range = pd.date_range(min_date,max_date,freq='D')
        cur_data = self.data[date_range]
        stats = self.get_stats(cur_data,decay=0.999)
        if prefix:
            stats = stats.add_prefix(prefix)
            
        self.cur_data = cur_data
        self.date_range = date_range
        return stats

class WeeklyMovingStats(TemporalMovingStats):
    def __init__(self,
                 data,
                 periods):
        self.data = data
        self.periods = periods
    
    def weekly_build(self,max_date):        
        ws = []
        for period in self.periods:
            weekly_stats = self.get_weekly_stats(max_date=max_date,period=period).add_prefix(f'period{period}_')
            ws.append(weekly_stats)
        ws = pd.concat(ws,axis=1)
        
        return ws

class OnpromoMovingStats(WeeklyMovingStats):
    def __init__(self,
                 data,
                 data_onpromo,
                 periods):
        super().__init__(data=data,
                         periods=periods)
        self.data_onpromo = data_onpromo
    
    def build(self,min_date,max_date,prefix=None):
        temp_stats = TemporalMovingStats.build(self,
                                               min_date=min_date,
                                               max_date=max_date,
                                               prefix=prefix)

        cur_data_onpromo = self.data_onpromo[self.date_range]
        onpromo_stats = self.get_onpromo_data(self.cur_data,cur_data_onpromo,decay=0.999)
        if prefix:
            onpromo_stats = onpromo_stats.add_prefix(prefix=prefix)
        
        stats = pd.concat([temp_stats,onpromo_stats],axis=1)
        return stats
        

#%%

    