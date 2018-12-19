# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 09:57:45 2018

@author: zhanglisama    jxufe
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
train = pd.read_table("train.txt")
test = pd.read_table("test.txt")
target = train.pop('target')
train['V2*V4'] = train['V2']*train['V4']
test['V2*V4'] = test['V2']*train['V4']
#train = train.drop(["V9","V14","V17","V19","V21","V22","V25","V26","V28","V29","V32","V33","V34","V35"],axis=1)
#train = train.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'],axis=1)
#test = test.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'],axis=1)
#train['V1*V1'] = train['V1']*train['V1']
#train['V0*V0'] = train['V0']*train['V0']
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


#方差
#threshold = 0                 
#vt = VarianceThreshold().fit(train)
## Find feature names
#feat_var_threshold = train.columns[vt.variances_ > threshold * (1-threshold)]
#train = train[feat_var_threshold]
#train_x,test_x,train_y,test_y = train_test_split(train,target,test_size=0.2,random_state=0)

n_folds = 5

#def msle_cv(model):
#    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x)
#    mse= -cross_val_score(model, train_x, train_y, scoring="neg_mean_squared_error", cv = kf)
#    return(mse)


lasso = make_pipeline(StandardScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(StandardScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


model_lgb = lgb.LGBMRegressor(objective='regression',
                              learning_rate=0.03, n_estimators=500,
                              )


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   

averaged_models = AveragingModels(models = ( ENet,KRR,GBoost,model_lgb,model_xgb))

averaged_models.fit(train,target)
pred = averaged_models.predict(test)

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功") 
text_save('submit.txt',pred)
#score = msle_cv(averaged_models)
#print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
#    def __init__(self, base_models, meta_model, n_folds=5):
#        self.base_models = base_models
#        self.meta_model = meta_model
#        self.n_folds = n_folds
#   
#    # We again fit the data on clones of the original models
#    def fit(self, X, y):
#        self.base_models_ = [list() for x in self.base_models]
#        self.meta_model_ = clone(self.meta_model)
#        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=0)
#        
#        # Train cloned base models then create out-of-fold predictions
#        # that are needed to train the cloned meta-model
#        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
#        for i, model in enumerate(self.base_models):
#            for train_index, holdout_index in kfold.split(X, y):
#                instance = clone(model)
#                self.base_models_[i].append(instance)
#                instance.fit(X[train_index], y[train_index])
#                y_pred = instance.predict(X[holdout_index])
#                out_of_fold_predictions[holdout_index, i] = y_pred
#                
#        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
#        self.meta_model_.fit(out_of_fold_predictions, y)
#        return self
#   
#    #Do the predictions of all base models on the test data and use the averaged predictions as 
#    #meta-features for the final prediction which is done by the meta-model
#    def predict(self, X):
#        meta_features = np.column_stack([
#            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
#            for base_models in self.base_models_ ])
#        return self.meta_model_.predict(meta_features)
#    
#stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
#                                                 meta_model = lasso)
#
#score = msle_cv(stacked_averaged_models)
#print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
#
#def msle(y, y_pred):
#    return mean_squared_error(y, y_pred)
#
#
#stacked_averaged_models.fit(train_x.values, train_y)
#stacked_train_pred = stacked_averaged_models.predict(train_x.values)
#stacked_pred = stacked_averaged_models.predict(test_x.values)
#print(msle(train_y, stacked_train_pred))
#









