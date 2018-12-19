40# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:41:29 2018

@author: zhanglisama    jxufe
"""

import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
# 读取数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submit = pd.read_csv('sample_submit.csv')
label_encoder = LabelEncoder()

today = pd.to_datetime(date(2018, 10, 12))
#
train['birth_date'] = pd.to_datetime(train['birth_date'])
train['age'] = (today - train['birth_date']).apply(lambda x: x.days) / 365.
positions = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk']
train['best_pos'] = train[positions].max(axis=1)
test['best_pos'] = test[positions].max(axis=1)
test['birth_date'] = pd.to_datetime(test['birth_date'])
test['age'] = (today - test['birth_date']).apply(lambda x: x.days) / 365.
train['BMI'] = 10000. * train['weight_kg'] / (train['height_cm'] ** 2)
test['BMI'] = 10000. * test['weight_kg'] / (test['height_cm'] ** 2)
train['is_gk'] = train['gk'] > 0
test['is_gk'] = test['gk'] > 0

def add_poly_features(data, column_names):
    features = data[column_names]
    rest_features = data.drop(column_names, axis=1)
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = pd.DataFrame(poly_transformer.fit_transform(features),
                                 columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1, col, poly_features[col])
    return rest_features

features = ['age','is_gk',
 'potential',
 'def',
 'ball_control',
 'reactions',
 'short_passing',
 'sho',
 'heading_accuracy',
 'finishing',
 'positioning','best_pos','height_cm', 'weight_kg',  'pac',
 'phy', 'international_reputation']
train_df = train[features]
test_df = test[features]
train_df = add_poly_features(train_df,['age','def','best_pos','potential'])
test_df = add_poly_features(test_df,['age','def','best_pos','potential'])
y = train.pop('y') 

#feature_names = np.array(train_df.columns)                                        


#x_train,x_test,y_train,y_test = train_test_split(train_df,y,test_size=0.1,random_state=0)
#model_xgb = xgb.XGBRegressor(n_estimators=1000)
#
#model_xgb.fit(x_train,y_train)
#
#predict = model_xgb.predict(x_test)
## 对测试集进行预测
#print('xgb mae :',mean_absolute_error(y_test,predict))
## 显示重要特征
#plot_importance(model_xgb)
#plt.show()
    
forest = RandomForestRegressor(random_state=100,n_estimators=2000)
#forest.fit(x_train, y_train)
##x_test = pre_data(test[features])
#pred = forest.predict(x_test)
#print('forest mae: ',mean_absolute_error(pred,y_test))
#plot_feature_importances(reg_ngk.feature_importances_, 
#            'feature importance', feature_names)
#test['pred'] =pred
#submit['y'] = np.array(test['pred'])
#submit.to_csv('result.csv', index=False)
#GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,
#                                   max_depth=4, max_features='sqrt',
#                                   min_samples_leaf=15, min_samples_split=10, 
#                                   loss='huber', random_state =5)
#GBoost.fit(x_train, y_train)
##x_test = pre_data(test[features])
#pred = GBoost.predict(x_test)
#print('GBoost mae: ',mean_absolute_error(pred,y_test))
lgbl = lgb.LGBMRegressor(n_estimators=2000,learning_rate=0.05)
#lgbl.fit(x_train,y_train)
#pred = lgbl.predict(x_test)
#print('lgbl mae: ',mean_absolute_error(pred,y_test))
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

averaged_models = AveragingModels(models = (lgbl,lgbl))

averaged_models.fit(train_df,y)
pred = averaged_models.predict(test_df)
#print('average mae: ',mean_absolute_error(pred,y_test))

test['pred'] =pred
submit['y'] = np.array(test['pred'])
submit.to_csv('result1.csv', index=False)

