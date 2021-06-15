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




x = np.arange(0,180,1)

random.seed(0)

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1,random_state = 120)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)


n_batch_size = 90
num_classes = 2
n_epochs = 100
input_shape=(x_train.shape[1], 1)

model = tensorflow.keras.models.load_model('D:\\MIA\\CE\\Proyecto Final\\Modelos\\8500.h5')

# model = Sequential()
# intput_shape=(x_train.shape[1], 1)
# model.add(Convolution1D(16, kernel_size=3, padding = 'same',activation='relu', input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=(2)))
# model.add(Convolution1D(32,kernel_size=3, padding = 'same',activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=(2)))
# model.add(Flatten())
# model.add(Dense(num_classes, activation='softmax'))


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train,y_train, batch_size = n_batch_size, epochs= n_epochs, verbose = 0)

_, accuracy = model.evaluate(x_test, y_test, batch_size=n_batch_size, verbose=0)
accuracy = round(accuracy  * 10000.00)
print('PRECISION: ', accuracy)
#model.save('D:\\MIA\\CE\\Proyecto Final\\Modelos\\'+str(accuracy)+'.h5')

# print(mat['__header__'])

# for i in range(200):
#     dataClass = mat['datam3'][i][1]
#     if dataClass == 5:
#         plt.plot(x,mat['datam3'][i][2:], label=str(i))
#         plt.legend()

# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('Coloscopy')
# plt.show()