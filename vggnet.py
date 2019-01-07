# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:16:33 2019

@author: zhanglisama    jxufe
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
from time import time
import cv2
from keras.models import load_model
from keras import models
from keras import layers
from keras import optimizers

#获取dataset
from keras.applications import VGG16
def load_data(dataset_path):
    
    X = []
    
    for dirname,dirnames,filensmes in os.walk(dataset_path):
        for subdirname in dirnames:   
            subject_path=os.path.join(dirname,subdirname)  
            for filename in os.listdir(subject_path):   
                im=cv2.imread(os.path.join(subject_path,filename))/256
                X.append(im)
                
    label = np.zeros((400, 40))
    for i in range(40):
        label[i * 10: (i + 1) * 10, i] = 1
        
   
    return  X,label
t0 = time()
dataset_path = r'F:\python\python-test\att_faces'
data,label = load_data(dataset_path)

data = np.reshape(data,(-1,112,92,3))

vgg = VGG16(weights='imagenet',include_top=False,input_shape=(112,92,3))

train_x,test_x,train_y,test_y = train_test_split(data,label,test_size=0.2)

train_x_feature = vgg.predict(train_x)
test_x_feature = vgg.predict(test_x)

train_x_feature = np.reshape(train_x_feature,(320,3*2*512))
test_x_feature = np.reshape(test_x_feature,(80,3*2*512))

model = models.Sequential()
model.add(layers.Dense(512,activation='relu',input_dim=3*2*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(40,activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss='categorical_crossentropy',metrics=['acc'])
history = model.fit(train_x_feature,train_y,epochs=90,batch_size=50,validation_split=0.1)

loss,acc = model.evaluate(test_x_feature ,test_y,batch_size= 20)
print('Test loss:',loss)
print('Test accuracy:', acc)

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,'b',label='Training loss')
plt.plot(epochs,val_loss_values,'r',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure()
#plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs,acc_values,'b',label='Training acc')
plt.plot(epochs,val_acc_values,'r',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()