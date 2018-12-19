# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:45:52 2018

@author: zhanglisama    jxufe
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
def get_hour(x):
    h = int(x[11:13])
    m = int(x[14:16])
    if m in [14, 29, 44]:
        m += 1
    if m == 59:
        m = 0
        h += 1
    if h == 24:
        h = 0
    return h * 60 + m


def add_poly_features(data, column_names):
    features = data[column_names]
    rest_features = data.drop(column_names, axis=1)
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = pd.DataFrame(poly_transformer.fit_transform(features),
                                 columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1, col, poly_features[col])
    return rest_features
train_x_old = pd.read_csv('train_1.csv')


def get_peak(df):
    nums = df['辐照度']
    peaks = []
    index=[]
    for idx in range(1, len(nums)-1):
        if nums[idx-1] < nums[idx] > nums[idx+1]:
            index.append(idx)
            peaks.append( nums[idx])
    return peaks,index
            
peaks,index = get_peak(train_x_old)


def get_dis2_peak(df):
    dis2=[]
    peak_values=[]   
    df['ID'] = list(range(len(df['辐照度'])))
    irr = df['辐照度']
    data = df
    ID=data['ID'] 
    for id in ID:
        mindis = np.abs(id-index[0])
        peak_row = index[0]
        for i,peak_id in enumerate(index):
            if np.abs(peak_id)<mindis:
                mindis = np.abs(id-peak_id)
                peak_row = index[i]
        dis2.append(mindis)
        peak_values.append(irr[peak_row])
    data =df
    data.insert(1,'distance2peak',dis2)
    mean_irr = []
    std_irr = []
    for dis in enumerate(data['distance2peak']):
        mean_irr_i = np.mean(data[data['distance2peak']==dis[1]]['辐照度'])
        std_irr_i=np.std(data[data['distance2peak']==dis[1]]['辐照度'])
        mean_irr.append(mean_irr_i)
        std_irr.append(std_irr_i)
    return dis2,peak_values,mean_irr,std_irr
dis2,peak_values,mean_irr,std_irr = get_dis2_peak(train_x_old)
train_x_old.insert(1,'peak_values',peak_values)
train_x_old.insert(1,'mean_irr',mean_irr)
train_x_old.insert(1,'std_irr',std_irr)
train_x_old.pop('ID')

##################################

def get_peak1(df):
    nums = df['温度']
    peaks = []
    index=[]
    for idx in range(1, len(nums)-1):
        if nums[idx-1] < nums[idx] > nums[idx+1]:
            index.append(idx)
            peaks.append( nums[idx])
    return peaks,index
            
peaks,index = get_peak1(train_x_old)


def get_dis2_peak1(df):
    dis2=[]
    peak_values=[]   
    df['ID'] = list(range(len(df['温度'])))
    irr = df['温度']
    data = df
    ID=data['ID'] 
    for id in ID:
        mindis = np.abs(id-index[0])
        peak_row = index[0]
        for i,peak_id in enumerate(index):
            if np.abs(peak_id)<mindis:
                mindis = np.abs(id-peak_id)
                peak_row = index[i]
        dis2.append(mindis)
        peak_values.append(irr[peak_row])
    data =df
    data.insert(1,'temp-distance2peak',dis2)
    mean_irr = []
    std_irr = []
    for dis in enumerate(data['temp-distance2peak']):
        mean_irr_i = np.mean(data[data['temp-distance2peak']==dis[1]]['温度'])
        std_irr_i=np.std(data[data['temp-distance2peak']==dis[1]]['温度'])
        mean_irr.append(mean_irr_i)
        std_irr.append(std_irr_i)
    return dis2,peak_values,mean_irr,std_irr
dis2,peak_values,mean_irr,std_irr = get_dis2_peak1(train_x_old)

train_x_old.insert(1,'temp-peak_values',peak_values)
train_x_old.insert(1,'temp-mean_irr',mean_irr)
train_x_old.insert(1,'temp-std_irr',std_irr)
train_x_old.pop('ID')

############################
def get_peak1(df):
    nums = df['湿度']
    peaks = []
    index=[]
    for idx in range(1, len(nums)-1):
        if nums[idx-1] < nums[idx] > nums[idx+1]:
            index.append(idx)
            peaks.append( nums[idx])
    return peaks,index
            
peaks,index = get_peak1(train_x_old)


def get_dis2_peak1(df):
    dis2=[]
    peak_values=[]   
    df['ID'] = list(range(len(df['湿度'])))
    irr = df['湿度']
    data = df
    ID=data['ID'] 
    for id in ID:
        mindis = np.abs(id-index[0])
        peak_row = index[0]
        for i,peak_id in enumerate(index):
            if np.abs(peak_id)<mindis:
                mindis = np.abs(id-peak_id)
                peak_row = index[i]
        dis2.append(mindis)
        peak_values.append(irr[peak_row])
    data =df
    data.insert(1,'hum-distance2peak',dis2)
    mean_irr = []
    std_irr = []
    for dis in enumerate(data['hum-distance2peak']):
        mean_irr_i = np.mean(data[data['hum-distance2peak']==dis[1]]['湿度'])
        std_irr_i=np.std(data[data['hum-distance2peak']==dis[1]]['湿度'])
        mean_irr.append(mean_irr_i)
        std_irr.append(std_irr_i)
    return dis2,peak_values,mean_irr,std_irr
dis2,peak_values,mean_irr,std_irr = get_dis2_peak1(train_x_old)

train_x_old.insert(1,'hum-peak_values',peak_values)
train_x_old.insert(1,'hum-mean_irr',mean_irr)
train_x_old.insert(1,'hum-std_irr',std_irr)
train_x_old.pop('ID')
###################################
def get_peak1(df):
    nums = df['风速']
    peaks = []
    index=[]
    for idx in range(1, len(nums)-1):
        if nums[idx-1] < nums[idx] > nums[idx+1]:
            index.append(idx)
            peaks.append( nums[idx])
    return peaks,index
            
peaks,index = get_peak1(train_x_old)


def get_dis2_peak1(df):
    dis2=[]
    peak_values=[]   
    df['ID'] = list(range(len(df['风速'])))
    irr = df['风速']
    data = df
    ID=data['ID'] 
    for id in ID:
        mindis = np.abs(id-index[0])
        peak_row = index[0]
        for i,peak_id in enumerate(index):
            if np.abs(peak_id)<mindis:
                mindis = np.abs(id-peak_id)
                peak_row = index[i]
        dis2.append(mindis)
        peak_values.append(irr[peak_row])
    data =df
    data.insert(1,'speed-distance2peak',dis2)
    mean_irr = []
    std_irr = []
    for dis in enumerate(data['speed-distance2peak']):
        mean_irr_i = np.mean(data[data['speed-distance2peak']==dis[1]]['风速'])
        std_irr_i=np.std(data[data['speed-distance2peak']==dis[1]]['风速'])
        mean_irr.append(mean_irr_i)
        std_irr.append(std_irr_i)
    return dis2,peak_values,mean_irr,std_irr
dis2,peak_values,mean_irr,std_irr = get_dis2_peak1(train_x_old)

train_x_old.insert(1,'speed-peak_values',peak_values)
train_x_old.insert(1,'speed-mean_irr',mean_irr)
train_x_old.insert(1,'speed-std_irr',std_irr)
train_x_old.pop('ID')

###########################

def get_peak1(df):
    nums = df['压强']
    peaks = []
    index=[]
    for idx in range(1, len(nums)-1):
        if nums[idx-1] < nums[idx] > nums[idx+1]:
            index.append(idx)
            peaks.append( nums[idx])
    return peaks,index
            
peaks,index = get_peak1(train_x_old)


def get_dis2_peak1(df):
    dis2=[]
    peak_values=[]   
    df['ID'] = list(range(len(df['压强'])))
    irr = df['压强']
    data = df
    ID=data['ID'] 
    for id in ID:
        mindis = np.abs(id-index[0])
        peak_row = index[0]
        for i,peak_id in enumerate(index):
            if np.abs(peak_id)<mindis:
                mindis = np.abs(id-peak_id)
                peak_row = index[i]
        dis2.append(mindis)
        peak_values.append(irr[peak_row])
    data =df
    data.insert(1,'pressure-distance2peak',dis2)
    mean_irr = []
    std_irr = []
    for dis in enumerate(data['pressure-distance2peak']):
        mean_irr_i = np.mean(data[data['pressure-distance2peak']==dis[1]]['压强'])
        std_irr_i=np.std(data[data['pressure-distance2peak']==dis[1]]['压强'])
        mean_irr.append(mean_irr_i)
        std_irr.append(std_irr_i)
    return dis2,peak_values,mean_irr,std_irr
dis2,peak_values,mean_irr,std_irr = get_dis2_peak1(train_x_old)

train_x_old.insert(1,'pressure-peak_values',peak_values)
train_x_old.insert(1,'pressure-mean_irr',mean_irr)
train_x_old.insert(1,'pressure-std_irr',std_irr)
train_x_old.pop('ID')
####################################

def get_peak1(df):
    nums = df['风向']
    peaks = []
    index=[]
    for idx in range(1, len(nums)-1):
        if nums[idx-1] < nums[idx] > nums[idx+1]:
            index.append(idx)
            peaks.append( nums[idx])
    return peaks,index
            
peaks,index = get_peak1(train_x_old)


def get_dis2_peak1(df):
    dis2=[]
    peak_values=[]   
    df['ID'] = list(range(len(df['风向'])))
    irr = df['风向']
    data = df
    ID=data['ID'] 
    for id in ID:
        mindis = np.abs(id-index[0])
        peak_row = index[0]
        for i,peak_id in enumerate(index):
            if np.abs(peak_id)<mindis:
                mindis = np.abs(id-peak_id)
                peak_row = index[i]
        dis2.append(mindis)
        peak_values.append(irr[peak_row])
    data =df
    data.insert(1,'dir-distance2peak',dis2)
    mean_irr = []
    std_irr = []
    for dis in enumerate(data['dir-distance2peak']):
        mean_irr_i = np.mean(data[data['dir-distance2peak']==dis[1]]['风向'])
        std_irr_i=np.std(data[data['dir-distance2peak']==dis[1]]['风向'])
        mean_irr.append(mean_irr_i)
        std_irr.append(std_irr_i)
    return dis2,peak_values,mean_irr,std_irr
dis2,peak_values,mean_irr,std_irr = get_dis2_peak1(train_x_old)

train_x_old.insert(1,'dir-peak_values',peak_values)
train_x_old.insert(1,'dir-mean_irr',mean_irr)
train_x_old.insert(1,'dir-std_irr',std_irr)
train_x_old.pop('ID')
##############################

test = pd.read_csv('test_1.csv')
train_x_old['month'] = train_x_old['时间'].apply(lambda x: x[5:7]).astype('int32')
train_x_old['day'] = train_x_old['时间'].apply(lambda x: x[8:10]).astype('int32')
train_x_old['hour'] = train_x_old['时间'].apply(lambda x: get_hour(x)).astype('int32')
train_x_old['the-hour'] = train_x_old['时间'].apply(lambda x: x[11:13]).astype('int32')
#train_x_old['irr-hour'] = train_x_old['irr']*train_x_old['the-hour']
#train_x_old.pop('the-hour')
test['month'] = test['时间'].apply(lambda x: x[5:7]).astype('int32')
test['day'] = test['时间'].apply(lambda x: x[8:10]).astype('int32')
test['hour'] = test['时间'].apply(lambda x: get_hour(x)).astype('int32')
test['the-hour'] = test['时间'].apply(lambda x: x[11:13]).astype('int32')

train_y = train_x_old['实际功率']

train_x = train_x_old.drop(['实际功率','实发辐照度'], axis=1)
train_x['dis2peak'] = train_x['hour'].apply(lambda x: (810 - abs(810 - x)) / 810)
train_x = add_poly_features(train_x, ['风速','风向','peak_values','dis2peak'])
train_x = add_poly_features(train_x, ['温度','湿度','压强','distance2peak'])

id = test['id']
#del_id = test[test['辐照度'].isin([-1.0])]['id']
#test = test.drop(['id'], axis=1)
#peaks,index = get_peak(test)
#dis2,peak_values,mean_irr,std_irr = get_dis2_peak(test)
#
#test.insert(1,'peak_values',peak_values)
#test.insert(1,'mean_irr',mean_irr)
#test.insert(1,'std_irr',std_irr)
#test.pop('ID')
#
#test['dis2peak'] = test['hour'].apply(lambda x: (810 - abs(810 - x)) / 810)
#test = add_poly_features(test, ['风速','风向','dis2peak','peak_values'])
#test = add_poly_features(test, ['温度','湿度','压强','distance2peak'])

#
train_x = train_x.drop(['时间'], axis=1)
test = test.drop(['时间'], axis=1)

print('train_x.shape,test_1.shape : ', train_x.shape, test.shape)

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=678)
#
params = {
    "objective": "regression",
    "metric": "mse",
    "num_leaves": 30,
    "min_child_samples": 100,
    "learning_rate": 0.03,
    "bagging_fraction": 0.7,
    "feature_fraction": 0.5,
    "bagging_frequency": 5,
    "bagging_seed": 666,
    "verbosity": -1
}
#
#
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
print('begin train')
gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=50000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100,
                    verbose_eval=100)
y_pred = gbm.predict(X_test)
#
###write result
##lgb_train = lgb.Dataset(train_x, label=train_y)
###lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
##print('begin train')
##gbm = lgb.train(params,
##                    lgb_train,
##                    num_boost_round=80000
##            )
##y_pred = gbm.predict(test)
##
#
#print('mae:',mean_absolute_error(y_pred,y_test))
#plt.figure()
#lgb.plot_importance(gbm,max_num_features=30)
#plt.show()
#gbm.feature_importance()


#republish_pred = gbm.predict(test)
#model = xgb.XGBRegressor(learning_rate=0.01,n_estimators=5000)
#
#model.fit(train_x,train_y)
#y_pred = model.predict(test)
#y_pred = pd.DataFrame(y_pred)
#sub = pd.concat([id, y_pred], axis=1)
##print(sub.shape)
sub.columns = ['id', 'predicition']
sub.loc[sub['id'].isin(del_id), 'predicition'] = 0.0
sub.to_csv('baseline.csv', index=False, sep=',', encoding='UTF-8')


#0.03  mae: 0.2526517947390355
##mae: 0.24454951918636356
##mae: 0.24162588753316336
###mae: 0.23898924671257937
###mae: 0.23805439178058424
###mae: 0.23587658488241894
####mae: 0.235342084005648
##### mae: 0.2323427749733568

##mae: 0.23162530163508852