# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:20:18 2018

@author: zhanglisama    jxufe
"""

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
train = pd.read_csv('train.csv',encoding='utf-8')
test = pd.read_csv('test.csv',encoding='utf-8')
submit = pd.read_csv('sample_submit.csv')
y = train.pop('y')

train.pop('id')
index = test.pop('id')
#feature_names = ['hour','temp_1','temp_2','is_workday']
#train1 = train[feature_names]

features= train.columns
def add_poly_features(data, column_names):
    features = data[column_names]
    rest_features = data.drop(column_names, axis=1)
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = pd.DataFrame(poly_transformer.fit_transform(features),
                                 columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1, col, poly_features[col])
    return rest_features

#train = add_poly_features(train,['temp_1','temp_2','hour','is_workday'])
#
x_train,x_test,y_train,y_test = train_test_split(train,y,test_size=0.4,random_state=0)
#
#forest = RandomForestRegressor(n_estimators=1000)
#forest.fit(x_train,y_train)
#pred = forest.predict(x_test)
#print('forest rmse:',sqrt(mean_squared_error(y_test,pred)))
GBoost = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
#model = xgb.XGBRegressor(learning_rate=0.1,n_estimators=1000)
#model.fit(x_train,y_train)
#pred2 = model.predict(x_test)
#print('xgb rmse',sqrt(mean_squared_error(y_test,pred2)))
#plot_importance(model)
#plt.show()

lgbl = lgb.LGBMRegressor(n_estimators=200,learning_rate=0.05)
lgbl.fit(x_train,y_train)
pred = lgbl.predict(x_test)
print('lgb rmse',sqrt(mean_squared_error(y_test,pred)))
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error


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
    
averaged_models = AveragingModels(models = ( GBoost,lgbl))
averaged_models.fit(train,y)
pred = averaged_models.predict(test)
#print('average rmse',sqrt(mean_squared_error(y_test,pred)))


#
test['pred'] =pred
submit['y'] = np.array(test['pred'])
submit.to_csv('result1.csv', index=False)
##








