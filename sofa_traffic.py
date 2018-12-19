# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 20:30:33 2018

@author: zhanglisama    jxufe
"""

import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from mlens.ensemble import SequentialEnsemble

from sklearn.metrics import average_precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
labels = train.pop('Evaluation')
train.drop('CaseId', axis=1, inplace=True)
test.drop('CaseId', axis=1, inplace=True)

x_train,x_test,y_train,y_test = train_test_split(train,labels,test_size=0.4,random_state=0)

lgbl = lgb.LGBMClassifier(n_estimators=200,learning_rate=0.05)
lgbl.fit(train,labels)
pred1 = lgbl.predict_proba(test)[:,1]
print('lgb ',average_precision_score(y_test,pred1))

gb = GradientBoostingClassifier(n_estimators=1000)
gb.fit(train,labels)
pred2 = gb.predict_proba(test)[:,1]
print('lgb ',average_precision_score(y_test,pred2))

rf = RandomForestClassifier(n_estimators=200)
rf.fit(x_train,y_train)
pred = rf.predict_proba(x_test)[:,1]
print('rf ',average_precision_score(y_test,pred))

xgb = xgb.XGBClassifier(n_estimators=1000)

xgb.fit(train,labels)
pred3 = xgb.predict_proba(test)[:,1]
print('xgb ',average_precision_score(y_test,pred2))
 
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
            model.predict_proba(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   

averaged_models = AveragingModels(models = (lgbl,xgb,gb))

averaged_models.fit(train,labels)
pred = averaged_models.predict(test)
print('average ',average_precision_score(y_test,pred))


ensemble = SequentialEnsemble()
ensemble.add('stack',[lgbl,xgb],proba=True)
ensemble.add('subsemble',[RandomForestClassifier(random_state=0,n_estimators=200),xgb],proba=True)
ensemble.add_meta(xgb)
ensemble.fit(x_train,y_train)
pred = ensemble.predict_proba(x_test)[:,1]
print('ensemble accuracy:',average_precision_score(pred,y_test))