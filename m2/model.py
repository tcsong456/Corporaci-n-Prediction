import lightgbm as lgb

class Lightgbm:
    def __init__(self,
                 params,
                 num_round,
                 weight=None):
        self.params = params
        self.num_round = num_round
        self.weight = weight
        
    def fit(self,X_tr,y_tr):
        print('Fitting')
        dtrain = lgb.Dataset(X_tr,label=y_tr,weight=self.weight)
        self.bst = lgb.train(self.params,dtrain,num_boost_round=self.num_round)
        return self
    
    def predict(self,X_te):
        pred = self.bst.predict(X_te,self.bst.best_iteration or self.num_round)
        return pred