# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:32:01 2021

@author: angel
"""

import scipy.io
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, BatchNormalization, Activation, Dropout, Flatten, Dense



random.seed(0)
num_classes = 2
# x_train = np.load('D:\\MIA\\CE\\Proyecto Final\\Modelos\\2 clases\\85xtrain.npy')
# x_test = np.load('D:\\MIA\\CE\\Proyecto Final\\Modelos\\2 clases\\85xtest.npy')
# y_train = np.load('D:\\MIA\\CE\\Proyecto Final\\Modelos\\2 clases\\85ytrain.npy')
# y_test = np.load('D:\\MIA\\CE\\Proyecto Final\\Modelos\\2 clases\\85ytest.npy')


mat = scipy.io.loadmat('D:\MIA\CE\Proyecto Final\Colposcopia\datam3.mat')

x = mat['datam3'][:,2:]
y = mat['datam3'][:,1]
for i in range(len(y)):
    if y[i] < 5:
        y[i] = 0
    else:
        y[i] = 1
    

x = np.array(x[:])
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1,random_state = 2)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)

input_shape=(x_train.shape[1], 1)

a_functionsC = ['sigmoid', 'tanh', 'relu', 'linear']
a_functionsFC = ['sigmoid', 'tanh', 'relu', 'softmax']
optimizers = ['adagrad', 'adadelta', 'adam', 'sgd']

def create_model(chromosome):
    n_epochs = int(chromosome[0:7], 2)
    n_batch_size = int(chromosome[7:15],2) + 1
    n_convolutional_layers = int(chromosome[15:17], 2) + 1
    model = Sequential()
    for i in range(n_convolutional_layers):
        m = 16*i
        n_filters = int(chromosome[17+m:23+m],2) + 1
        k_size = int(chromosome[23+m:27+m],2) + 1
        activation_f = int(chromosome[27+m:29+m],2)
        max_pool = int(chromosome[29+m:30+m], 2)
        tam_pool = int(chromosome[30+m:33+m],2) + 1        
        model.add(Convolution1D(n_filters, kernel_size=k_size, padding = 'same',activation=a_functionsC[activation_f], input_shape=input_shape)) 
        if max_pool == 1:
            model.add(MaxPooling1D(pool_size=(tam_pool)))
    m = 48
    model.add(Flatten())
    n_fc_layers = int(chromosome[33+m:35+m],2) + 1
    for j in range(n_fc_layers):
        n = 9 * j
        n_neurons = int(chromosome[35+m+n:42+m+n],2) + 1
        activation_f = int(chromosome[42+m+n:44+m+n], 2)
        
        model.add(Dense(n_neurons, activation=a_functionsFC[activation_f]))        
    
    n = 27
    model.add(Dense(num_classes, activation='softmax'))            
    opt = int(chromosome[44+m+n:46+m+n],2)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers[opt], metrics=['accuracy'])
    model.fit(x_train,y_train, batch_size = n_batch_size, epochs= n_epochs, verbose = 0)    
    _, accuracy = model.evaluate(x_test, y_test, batch_size=n_batch_size, verbose=0)
    
    accuracy = round(accuracy  * 10000.00)
    model.summary()
    return accuracy, model
    
    
modelo = '1100100010110010111111100101010010111110100101001000000000000000000000000000000000111111110101111110100111110100011110110'
acc, model = create_model(modelo)
print(len(modelo))
print(acc)

# model = tensorflow.keras.models.load_model('D:\\MIA\\CE\\Proyecto Final\\Modelos\\8000.h5')
# n_epochs = 100
# n_batch_size = 90
# input_shape=(x_train.shape[1], 1)
# model = Sequential()
# model.add(Convolution1D(16, kernel_size=3, padding = 'same',activation='relu', input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=(2)))
# model.add(Convolution1D(32,kernel_size=3, padding = 'same',activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=(2)))
# model.add(Flatten())
# model.add(Dense(num_classes, activation='softmax'))

# print('aqui 1')
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print('aqui 2')
# model.fit(x_train,y_train, batch_size = n_batch_size, epochs= n_epochs, verbose = 0)
# print('aqui 3')
# _, accuracy = model.evaluate(x_test, y_test, batch_size=n_batch_size, verbose=0)
# accuracy = round(accuracy  * 10000.00)
# print('PRECISION: ', accuracy)
# model.save('D:\\MIA\\CE\\Proyecto Final\\Modelos\\'+str(accuracy)+'.h5')
