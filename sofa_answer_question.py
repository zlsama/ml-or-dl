# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:34:12 2018

@author: zhanglisama    jxufe
"""


import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from mlens.ensemble import SequentialEnsemble
from sklearn.model_selection import KFold
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import LinearRegression
import featuretools as ft
from sklearn.preprocessing import PolynomialFeatures
train = pd.read_csv(r'train.csv',encoding='utf-8')
test = pd.read_csv(r'test.csv',encoding='utf-8')

def add_poly_features(data, column_names):
    features = data[column_names]
    rest_features = data.drop(column_names, axis=1)
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = pd.DataFrame(poly_transformer.fit_transform(features),
                                 columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1, col, poly_features[col])
    return rest_features

def error(y,y_pred):
    error = (np.sum(np.abs(y-y_pred)/y))/len(y)
    return error

cols_lr = ['id', 'sqrt_id']
train['sqrt_id'] = np.sqrt(train['id'])
test['sqrt_id'] = np.sqrt(test['id'])

   
# 构造星期、月、年特征
train['day'] = train['date'].apply(lambda x: x[-2:]).astype('int32')
train['date'] = pd.to_datetime(train['date'])
#train['quarter'] = train['date'].dt.quarter
train['d_w'] = train['date'].dt.dayofweek
train['week'] = train['date'].dt.weekofyear
train['dayofyear'] = train['date'].dt.dayofyear
train['d_m'] = train['date'].dt.month
train['d_y'] = train['date'].dt.year
train['weekend']=train['d_w'].apply(lambda x=1:x==5 or x==6)
train = add_poly_features(train,['d_w','weekend'])

test['day'] = test['date'].apply(lambda x: x[-2:]).astype('int32')
test['date'] = pd.to_datetime(test['date'])
test['d_w'] = test['date'].dt.dayofweek
test['week'] = test['date'].dt.weekofyear
test['dayofyear'] = test['date'].dt.dayofyear
test['d_m'] = test['date'].dt.month
test['d_y'] = test['date'].dt.year
test['weekend']=test['d_w'].apply(lambda x=1:x==5 or x==6)
test = add_poly_features(test,['d_w','weekend'])
cols_knn = ['d_w', 'd_m', 'd_y']

questions = train.pop('questions')
answer = train.pop('answers')


train = train.drop(['date'],axis=1)
test = test.drop(['date'],axis=1)
train_x,test_x,train_y,test_y = train_test_split(train,questions,test_size=0.1,random_state=0)

# 根据特征['id', 'sqrt_id']，构造线性模型预测questions
reg = LinearRegression()
reg.fit(train_x, train_y)
q_fit = reg.predict(train_x)
q_pred = reg.predict(test_x)

model = xgb.XGBRegressor(learning_rate=0.1,n_estimators=1600)
model.fit(train_x,train_y)
pred1 = model.predict(test_x)
print('xgb mape',error(test_y,pred1))

lgbl = lgb.LGBMRegressor(n_estimators=500,learning_rate=0.1)
lgbl.fit(train_x,train_y)
pred2 = lgbl.predict(test_x)
print('lgbl mape',error(test_y,pred2))
# 根据特征['id', 'sqrt_id']，构造线性模型预测answers
#reg = LinearRegression()
#reg.fit(train[cols_lr], train['answers'])
#a_fit = reg.predict(train[cols_lr])
#a_pred = reg.predict(test[cols_lr])

# 得到questions和answers的训练误差
q_diff = train_y - q_fit
#a_diff = train['answers'] - a_fit

# 把训练误差作为新的目标值，使用特征cols_knn，建立kNN模型
from sklearn.neighbors import KNeighborsRegressor
reg = KNeighborsRegressor()
reg.fit(train_x, q_diff)
q_pred_knn = reg.predict(test_x)
reg = KNeighborsRegressor()
#reg.fit(train[cols_knn], a_diff)
#a_pred_knn = reg.predict(test[cols_knn])
pred = q_pred_knn+q_pred
print('error',error(pred,test_y))
#输出预测结果至my_Lr_Knn_prediction.csv
#submit['questions'] = q_pred + q_pred_knn
#submit['answers'] = a_pred + a_pred_knn
#submit.to_csv('my_Lr_Knn_prediction.csv', index=False)
