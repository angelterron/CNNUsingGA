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
import gc
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, BatchNormalization, Activation, Dropout, Flatten, Dense


nFile = open("corridas6Clases.txt","w+")

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
for z in range(5):
    gc.collect()
    
num_classes = 6

# x_train = np.load('D:\\MIA\\CE\\Proyecto Final\\Modelos\\2 clases\\85xtrain.npy')
# x_test = np.load('D:\\MIA\\CE\\Proyecto Final\\Modelos\\2 clases\\85xtest.npy')
# y_train = np.load('D:\\MIA\\CE\\Proyecto Final\\Modelos\\2 clases\\85ytrain.npy')
# y_test = np.load('D:\\MIA\\CE\\Proyecto Final\\Modelos\\2 clases\\85ytest.npy')


mat = scipy.io.loadmat('D:\MIA\CE\Proyecto Final\Colposcopia\datam3.mat')

x = mat['datam3'][:,2:]
y = mat['datam3'][:,1]

for i in range(len(y)):
    y[i] = y[i]-1    
    

x = np.array(x[:])
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state = 235)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)

input_shape=(x_train.shape[1], 1)

a_functionsC = ['sigmoid', 'tanh', 'relu', 'linear']
a_functionsFC = ['sigmoid', 'tanh', 'relu', 'softmax']
optimizers = ['adagrad', 'adadelta', 'adam', 'sgd']

def imprimir(X, g):
    
    nFile.write('==== Gen ' + str(g) +' ====\n')
    for i in range(len(X)):        
        nFile.write(str(X[i][0]) + 'Accuracy: ' + str(X[i][1]/100.00) + '\n\n')

def key_solutions(elem):
    return elem[1]

def definirVe(poblacion):
    prom = 0
    
    for x in poblacion:
        prom += x[1]
    
    prom = prom/population_size
    acum = 0    
    
    for i in range(population_size):
        poblacion[i].append(poblacion[i][1]/prom)
        acum += poblacion[i][1]/prom   
        
def muta(offspring):
    inf = random.randint(0, len(offspring))
    sup = random.randint(0, len(offspring))
    
    while inf > sup:
        sup = random.randint(0, len(offspring))
    
    sub = offspring[inf:sup]
    offspring = offspring[0:inf] + sub[::-1] + offspring[sup:]

def create_model(chromosome):
    n_epochs = int(chromosome[0:7], 2)
    n_batch_size = int(chromosome[7:15],2) + 1
    n_convolutional_layers = int(chromosome[15:17], 2) + 1
    model = Sequential()
    for i in range(n_convolutional_layers):
        
        m = 15*i
        n_filters = int(chromosome[17+m:23+m],2) + 1
        k_size = int(chromosome[23+m:26+m],2) + 1
        activation_f = int(chromosome[26+m:28+m],2)
        max_pool = int(chromosome[28+m:29+m], 2)
        tam_pool = int(chromosome[29+m:32+m],2) + 1        
        if i == 0:
            model.add(Convolution1D(n_filters, kernel_size=k_size, padding = 'same',activation=a_functionsC[activation_f], input_shape=input_shape)) 
        else:
            model.add(Convolution1D(n_filters, kernel_size=k_size, padding = 'same',activation=a_functionsC[activation_f])) 
        if max_pool == 1:
            model.add(MaxPooling1D(pool_size=(tam_pool), padding='same'))
    m = 45
    model.add(Flatten())
    n_fc_layers = int(chromosome[32+m:34+m],2) + 1
    for j in range(n_fc_layers):
        n = 9 * j
        n_neurons = int(chromosome[34+m+n:41+m+n],2) + 1
        activation_f = int(chromosome[41+m+n:43+m+n], 2)
        
        model.add(Dense(n_neurons, activation=a_functionsFC[activation_f]))        
    
    n = 27
    model.add(Dense(num_classes, activation='softmax'))            
    opt = int(chromosome[43+m+n:45+m+n],2)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers[opt], metrics=['accuracy'])
    model.fit(x_train,y_train, batch_size = n_batch_size, epochs= n_epochs, verbose = 0)    
    _, accuracy = model.evaluate(x_test, y_test, batch_size=n_batch_size, verbose=0)
    
    accuracy = round(accuracy  * 10000.00)
    # model.summary()
    # print('Precision: ', accuracy)
    return accuracy, model
for r in range(15):
    print('\n\n\n====== ITERACION ' + str(r) + ' =======\n')
    nFile.write('\n\n\n====== ITERACION ' + str(r) + ' =======\n')
    population_size = 20
    population = []
    chromosomeL = 117
    generations = 25
    paramCrossover = 0.5
    paramMutation = 0.3
    g = 0
        
    
    for i in range(population_size):        
        pools = []    
        kernels = []
        # Epochs
        chromosome = '{0:07b}'.format(random.randint(0, 127))
        # Batch size
        chromosome += '{0:08b}'.format(random.randint(0, 255))
        # Convolutional Layers
        chromosome += '{0:02b}'.format(random.randint(0, 3))
        # Creation of pools
        for i in range(4):
            new_number = random.randint(0, 7) 
            while new_number in pools:
                new_number = random.randint(0, 7) 
            pools.append(new_number)
        
        pools.sort(reverse=True)
        
        for i in range(4):
            new_number = random.randint(0, 7)         
            pools.append(new_number)
        
        kernels.sort(reverse=True)
        
        # Convolutional Layers
        for i in range(4):
            chromosome += '{0:06b}'.format(random.randint(0, 63))    
            chromosome += '{0:03b}'.format(random.randint(0, 7))    
            chromosome += '{0:02b}'.format(random.randint(0, 3))    
            chromosome += '{0:01b}'.format(random.randint(0, 1))    
            chromosome += '{0:03b}'.format(pools[i])
        
        chromosome += '{0:02b}'.format(random.randint(0, 3))
        
        
        # Fully Connected Layers
        for i in range(4):
            chromosome += '{0:07b}'.format(random.randint(0, 127))            
            chromosome += '{0:02b}'.format(random.randint(0, 3))    
        
        # Optimizer
        chromosome += '{0:02b}'.format(random.randint(0, 3))
            
            
        # modelo = []
        # n_epochs = int(chromosome[0:7], 2)
        # n_batch_size = int(chromosome[7:15],2) + 1
        # n_convolutional_layers = int(chromosome[15:17], 2) + 1
        # modelo.append(n_epochs)
        # modelo.append(n_batch_size)
        # modelo.append(n_convolutional_layers)
        
        # for i in range(n_convolutional_layers):
        #     m = 16*i
        #     n_filters = int(chromosome[17+m:23+m],2) + 1
        #     k_size = int(chromosome[23+m:27+m],2) + 1
        #     activation_f = int(chromosome[27+m:29+m],2)
        #     max_pool = int(chromosome[29+m:30+m], 2)
        #     tam_pool = int(chromosome[30+m:33+m],2) + 1      
        #     modelo.append([n_filters,k_size,activation_f, max_pool, tam_pool])        
        # m = 48    
        # n_fc_layers = int(chromosome[33+m:35+m],2) + 1
        # modelo.append(n_fc_layers)
        # for j in range(n_fc_layers):
        #     n = 9 * j
        #     n_neurons = int(chromosome[35+m+n:42+m+n],2) + 1
        #     activation_f = int(chromosome[42+m+n:44+m+n], 2)        
        #     modelo.append([n_neurons,activation_f])
            
        # n = 27    
        # opt = int(chromosome[44+m+n:46+m+n],2)
        # modelo.append(opt)
        
        # print(modelo)
        acc, modelGA = create_model(chromosome)
        population.append([chromosome, acc, modelGA])
    
    
        
    while g < generations:
        definirVe(population)    
        parents = []
        ptr = random.random()
        
        suma = 0    
        random.shuffle(population)
        for i in range(population_size):  
            
            terminado = False
            suma += population[i][3]
            while terminado is False:             
                if suma > ptr:
                    parents.append(population[i])            
                    ptr += 1                
                else:
                    terminado = True
        
        
        random.shuffle(parents)    
        offsprings = []
        count = 0        
        while len(offsprings) < population_size:    
            
            x1 = parents[count]
            x2 = parents[count+1]        
            if random.random() < paramCrossover:            
                
                offspring = ''
                for i in range(len(x1[0])):
                    if random.random() < 0.5:
                        offspring+=x1[0][i]
                    else:
                        offspring+=x2[0][i]        
                
                y1 = offspring            
                
                offspring = ''
                for i in range(len(x1[0])):
                    if random.random() < 0.5:
                        offspring+=x1[0][i]
                    else:
                        offspring+=x2[0][i]
                
                
                y2 = offspring
                
            else:
                y1 = x1[0]
                y2 = x2[0]
                
            if random.random() < paramMutation:
                muta(y1)
                
            if random.random() < paramMutation:
                muta(y2)
            
            acc, modelGA = create_model(y1)
            offsprings.append([y1, acc, modelGA])
            
            acc, modelGA = create_model(y2)                
            offsprings.append([y2, acc, modelGA])
            
            count+=2
        g+=1
        
        population.sort(key=key_solutions, reverse=True)
        offsprings.sort(key=key_solutions, reverse=True)
        
        offsprings = offsprings[:population_size-1]
        offsprings.append([population[0][0], population[0][1],population[0][2]])
        population = offsprings
        population.sort(key=key_solutions, reverse=False)
        imprimir(population, g)
        print('\n\n\n====== GENERACION ' + str(g) + ' =======\n')

nFile.close()
    
