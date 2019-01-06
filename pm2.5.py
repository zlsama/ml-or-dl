# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:43:41 2019

@author: zhanglisama    jxufe
"""

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
train = pd.read_csv('pm25_train.csv')
test = pd.read_csv('pm25_test.csv')
label = train.pop('pm2.5')

train['date'] = pd.to_datetime(train['date'])
train['day'] = train['date'].dt.day
#train['quarter'] = train['date'].dt.quarter
train['d_w'] = train['date'].dt.dayofweek
train['week'] = train['date'].dt.weekofyear
train['dayofyear'] = train['date'].dt.dayofyear
train['d_m'] = train['date'].dt.month
train['d_y'] = train['date'].dt.year
train['weekend']=train['d_w'].apply(lambda x=1:x==5 or x==6)

test['date'] = pd.to_datetime(test['date'])
test['day'] = test['date'].dt.day

test['d_w'] = test['date'].dt.dayofweek
test['week'] = test['date'].dt.weekofyear
test['dayofyear'] = test['date'].dt.dayofyear
test['d_m'] = test['date'].dt.month
test['d_y'] = test['date'].dt.year
test['weekend']=test['d_w'].apply(lambda x=1:x==5 or x==6)

train = train.drop('date',axis=1)
test = test.drop('date',axis=1)


def add_poly_features(data, column_names):
    features = data[column_names]
    rest_features = data.drop(column_names, axis=1)
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = pd.DataFrame(poly_transformer.fit_transform(features),
                                 columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1, col, poly_features[col])
    return rest_features

train = add_poly_features(train,['dayofyear',"PRES",'day'])
test = add_poly_features(test,['dayofyear',"PRES",'day'])
train_x,test_x,train_y,test_y = train_test_split(train,label,test_size=0.2,random_state=0)

#xgbl = xgb.XGBRegressor(learning_rate=0.05,n_estimators=2000)
#xgbl.fit(train_x,train_y)
#pred = xgbl.predict(test_x)
#mse1=mean_squared_error(pred,test_y)
#print('mse:',mse1)
#print('mse:',round(mse1,2))

#lgbl = lgb.LGBMRegressor(n_estimators=2000,learning_rate=0.1)
#lgbl.fit(train_x,train_y)
#pred1 = lgbl.predict(test_x)
#mse=mean_squared_error(pred1,test_y)
#print('mse:',mse)
#print('mse:',round(mse,2))
#lgb.plot_importance(lgbl)

lgbl = lgb.LGBMRegressor(n_estimators=2000,learning_rate=0.1)
lgbl.fit(train,label)
pred = lgbl.predict(test)
dataframe = pd.DataFrame({'pm2.5': pred})
dataframe.to_csv("pm.csv", index=False, encoding='utf-8')