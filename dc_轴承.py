# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 17:15:18 2018

@author: zhanglisama    jxufe
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.fftpack import fft
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier
from mlens.ensemble import SequentialEnsemble
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier


data = pd.read_csv(r'train.csv',encoding='utf-8').values
test = pd.read_csv(r'test_data.csv',encoding='utf-8').values[:,1:]
train = data[:,1:6001]
labels = np.array(data[:,-1],dtype=np.int)

x_train= fft(train).real

train_x,test_x,train_y,test_y = train_test_split(x_train,labels,test_size=0.2,random_state=0)

train_x = np.reshape(train_x,(train_x.shape[0],1,train_x.shape[1]))
test_x = np.reshape(test_x,(test_x.shape[0],1,test_x.shape[1]))


ensemble = SequentialEnsemble()
ensemble.add('blend',[AdaBoostClassifier(n_estimators=1000,learning_rate=0.05),RandomForestClassifier(random_state=0,n_estimators=1000)])
ensemble.add('stack',[AdaBoostClassifier(n_estimators=1000,learning_rate=0.05),RandomForestClassifier(random_state=0,n_estimators=1000)])
ensemble.add('subsemble',[AdaBoostClassifier(n_estimators=1000,learning_rate=0.05),RandomForestClassifier(random_state=0,n_estimators=1000)])
ensemble.add_meta(XGBClassifier(learning_rate=0.05,n_estimators=1000,random_state=0))
ensemble.fit(train_x,train_y)
preds = ensemble.predict(test_x)
print('xgb accuracy:',accuracy_score(preds,test_y))
params = {
    "objective": "multiclass",
  
    "learning_rate": 0.03,
    'num_class':10
}

lgb_train = lgb.Dataset(train_x, label=train_y)
print('begin train')
gbm = lgb.train(params,
                    lgb_train
                    )
y_pred = gbm.predict(test_x)

pred = np.argmax(y_pred,axis=1)
print('xgb accuracy:',accuracy_score(pred,test_y))
lgbl = lgb.LGBMClassifier(learning_rate=0.03,n_estimators=1500)
lgbl.fit(x_train,labels)

x_test= fft(test).real
pred = lgbl.predict(x_test)
params = {
    "objective": "multiclass",
  
    "learning_rate": 0.03,
    'num_class':10
}



lgb_train = lgb.Dataset(x_train, label=labels)

print('begin train')
gbm = lgb.train(params,
                    lgb_train
                    )
y_pred = gbm.predict(x_test)

pred = np.argmax(y_pred,axis=1)

estimator = lgb.sklearn.LGBMClassifier(learning_rate=0.03,n_estimators=1000)
estimator.fit(train,labels)
pred = estimator.predict(test)

ensemble = SequentialEnsemble()
ensemble.add('stack',[RandomForestClassifier(random_state=0,n_estimators=1000),GaussianNB()],proba=True)
ensemble.add('subsemble',[RandomForestClassifier(random_state=0,n_estimators=1000),GaussianNB()],proba=True)
ensemble.add_meta(XGBClassifier(n_estimators=1000,random_state=0))
ensemble.fit(train_x,train_y)
pred = ensemble.predict(test_x)
print('xgb accuracy:',accuracy_score(pred,test_y))
pred = np.array(pred,dtype=np.int)
#accuracy_score(test,preds)
index = np.array(range(1,529))
dataframe = pd.DataFrame({'id':index,'label': pred})
dataframe.to_csv("predict.csv", index=False, encoding='utf-8',header=True)

pred = pd.DataFrame(pred)
sub = pd.concat([index, pred], axis=1)
print(sub.shape)
sub.columns = ['id', 'label']
#sub.loc[sub['id'].isin(del_id), 'predicition'] = 0.0
sub.to_csv('submit11.csv', index=False, sep=',', encoding='UTF-8')
