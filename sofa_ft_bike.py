# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:07:55 2018

@author: zhanglisama    jxufe
"""

import pandas as pd
import numpy as np
import featuretools as ft
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from mlens.ensemble import SequentialEnsemble
from xgboost import XGBRegressor


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
label = train.pop('y')
combi = train.append(test,ignore_index=True)

es = ft.EntitySet()
es.entity_from_dataframe(entity_id='bike',dataframe=combi,index='id')

es = es.normalize_entity(base_entity_id='bike',new_entity_id='ev',index='id',
                         additional_variables=['temp_1','temp_2'])
feature_matrix, feature_names = ft.dfs(entityset=es, 
target_entity = 'bike', 
max_depth = 2, 
verbose = 1)

feature_matrix = feature_matrix.reindex(index=combi['id'])
feature_matrix = feature_matrix.reset_index().dropna(axis=1,how='any')
feature_matrix.pop('id')
train_data = feature_matrix[:10000]
test_data = feature_matrix[10000:]


x_train,x_test,y_train,y_test = train_test_split(train_data,label,test_size=0.4,random_state=0)

lgbl = lgb.LGBMRegressor(n_estimators=200,learning_rate=0.05)
lgbl.fit(x_train,y_train)
pred = lgbl.predict(x_test)
print('lgb rmse',sqrt(mean_squared_error(y_test,pred)))

rf = RandomForestRegressor(n_estimators=200)
rf.fit(x_train,y_train)
pred = rf.predict(x_test)
print('rf rmse',sqrt(mean_squared_error(y_test,pred)))

xgb = XGBRegressor(n_estimators=1000)

xgb.fit(x_train,y_train)
pred = xgb.predict(x_test)
print('xgb rmse',sqrt(mean_squared_error(y_test,pred)))

ensemble = SequentialEnsemble()
#ensemble.add('blend',[lgbl,rf])
ensemble.add('stack',[lgbl,xgb])
ensemble.add('subsemble',[lgbl,rf])
ensemble.add_meta(lgbl)
ensemble.fit(x_train,y_train)
pred = ensemble.predict(x_test)
print('ensemble rmse',sqrt(mean_squared_error(y_test,pred)))



