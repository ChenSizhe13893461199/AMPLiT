# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:33:20 2024

@author: Sizhe Chen
"""
#The following part will introduce the required packages
import os
import string
import torch
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalAveragePooling1D

#from keras.layers import Input, merge, Flatten
from keras.layers import Input
from keras.layers.reshaping import Flatten
from keras.layers import concatenate, add
from keras.layers.normalization import batch_normalization
from keras.layers import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras.layers import Conv1D, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from utils1 import getMatrixLabel, Phos1,getMatrixLabelFingerprint, getMatrixLabelh, getMatrixLabelnlp
from keras.optimizers import adam_v2
#from utils1 import channel_attenstion
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import csv
import numpy as np
import keras.utils.np_utils as kutils
from keras.optimizers import adam_v2
from keras.layers import Conv1D, Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
#from keras.layers import Input, merge, Flatten
from keras.layers import Input
from keras.layers.reshaping import Flatten
from keras.layers import concatenate, add
from keras.models import Sequential, Model
import numpy as np
import keras.utils.np_utils as kutils
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.PyPro import GetProDes
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.GetProteinFromUniprot import GetProteinSequence as gps
from propy import GetSubSeq
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.AAComposition import CalculateAAComposition
from propy.AAComposition import CalculateAADipeptideComposition
from propy.AAComposition import GetSpectrumDict
from propy.AAComposition import Getkmers
from sklearn.preprocessing import StandardScaler

#Constructing the input dataset (sequential one-hot code)
train_file_name ='TrainingAMP.csv'  # Training dataset
win1 = 50

X1, T,rawseq, length = getMatrixLabelh(train_file_name, win1)
train_file_name = 'Non-AMPsfilter.csv'  # Test dataset

win1 = 50
X1tt, y_train1,rawseq1, length = getMatrixLabelh(train_file_name, win1)

train_file_name = 'Validation.csv' #Validation dataset
win1 = 50
X_val, y_train111,rawseq116, length = getMatrixLabelh(train_file_name, win1)




#Constructing the input dataset (physiochemical descriptors)
#These descriptors can also be obtained by codes: 

    #train_file_name = 'TrainingAMP.csv'  # Training dataset
    # win1 = 50
    #X, T, rawseq, length = getMatrixLabelFingerprint(train_file_name, win1)


X2 = np.load(file="Training_vector.npy")# Descriptor of Training dataset
X2tt = np.load(file="Test_vector.npy")# Descriptor of Test dataset
X2_val = np.load(file="5810_vector.npy")# Descriptor of Validation dataset

#The q,q1,z were originally created for training data split. For users, you can define it by your own
q=5810
q1=0
z=0
#43404+16000+300
aaa = np.zeros((43404+z+q1+q, 50, 20))
bbb = np.zeros((43404+z+q1+q, 91, 17))
aaa[:43404] = X1[:]
aaa[43404:43404+z] = X1tt[:z]
aaa[43404+z:43404+z+q1] = X1tt[990:990+q1]
aaa[43404+z+q1:43404+z+q1+q] = X_val[:]

bbb[:43404] = X2[:]
bbb[43404:43404+z] = X2tt[:z]
bbb[43404+z:43404+z+q1] = X2tt[990:990+q1]
bbb[43404+z+q1:43404+z+q1+q] = X2_val[:]



ddd = np.zeros(shape=(43404+z+q1+q, 2))
ddd[:43404,0:2] = T[:]
ddd[43404:43404+z,0:2]=y_train1[:z]
ddd[43404+z:43404+z+q1,0:2]=y_train1[z:z+q1]
ddd[43404+z+q1:43404+z+q1+q,0:2] = y_train111[:]

zzz1=np.zeros((39253-z-q1, 50, 20))
zzz2=np.zeros((39253-z-q1, 91, 17))

zzz1[:990-z]=X1tt[z:990]
zzz1[990-z:]=X1tt[990+q1:]

zzz2[:990-z]=X2tt[z:990]
zzz2[990-z:]=X2tt[990+q1:]

dddz = np.zeros(shape=(39253-z-q1, 2))
dddz[:990-z]=y_train1[z:990]
dddz[990-z:]=y_train1[990+q1:]


#Calculating the word2vec matrixes
import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
###################################################################
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format('word2vec11.bin', binary=True)

vector_size=100 #the length of the vector
ccc = np.zeros((len(rawseq), 50, vector_size))
for i in range(0,len(rawseq)):
    t=0
    for AA in rawseq[i]:
        ccc[i][t][:]=wv[AA]
        t=t+1
    for x in range(0,50-len(rawseq[i])):
        ccc[i][t][:]=-10**(-100000)
        t=t+1
    #print(i)
    
ccc1 = np.zeros((len(rawseq116), 50, vector_size))
for i in range(0,len(rawseq116)):
    t=0
    for AA in rawseq116[i]:
        ccc1[i][t][:]=wv[AA]
        t=t+1
    for x in range(0,50-len(rawseq116[i])):
        ccc1[i][t][:]=-10**(-100000)
        t=t+1
    #print(i)

ccc2 = np.zeros((len(rawseq1), 50, vector_size))
for i in range(0,len(rawseq1)):
    t=0
    for AA in rawseq1[i]:
        ccc2[i][t][:]=wv[AA]
        t=t+1
    for x in range(0,50-len(rawseq1[i])):
        ccc2[i][t][:]=-10**(-100000)
        t=t+1
    #print(i)

vector=np.zeros((len(rawseq)+z+q1+q, 50, vector_size))

vector[:43404] = ccc[:]
vector[43404:43404+z] = ccc2[:z]
vector[43404+z:43404+z+q1] = ccc2[990:990+q1]
vector[43404+z+q1:43404+z+q1+q] = ccc1[:]

zzz3=np.zeros((39253-z-q1, 50, vector_size))
zzz3[:990-z]=ccc2[z:990]
zzz3[990-z:]=ccc2[990+q1:]


#model training
img_dim1 = aaa.shape[1:]

img_dim2 = vector.shape[1:]

img_dim3 = bbb.shape[1:]

img_dim4 = bbb.shape[1:]

img_dim5 = bbb.shape[1:]

img_dim6 = bbb.shape[1:]

init_form = 'RandomUniform'
learning_rate = 0.0001 #0.001
nb_dense_block =9
nb_layers = 9
nb_filter = 16 #16
growth_rate = 16 #16
filter_size_block1 = 15 #15
filter_size_block2 = 75 #75
filter_size_block3 = 15 #15
filter_size_block4 = 0 #abandoned
filter_size_block5 = 0 #abandoned
filter_size_block6 = 0 #abandoned
filter_size_ori = 1
dense_number = 16 #16
dropout_rate = 0.2
dropout_dense = 0.2
weight_decay = 0.000001
nb_batch_size = 512 #batch size
nb_classes = 2 #2 classes
nb_epoch = 21 #epochs
file_name='TrainingAMP.csv'

model1,parameter = Phos1(nb_classes, nb_layers, img_dim1, img_dim2, img_dim3,img_dim4,img_dim5,img_dim6, init_form, nb_dense_block,growth_rate, 
               filter_size_block1, filter_size_block2, filter_size_block3,filter_size_block4,filter_size_block5,filter_size_block6,
               nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay)

opt = adam_v2.Adam(learning_rate=learning_rate,
                   beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model1.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

history = model1.fit([aaa[:43404],vector[:43404],bbb[:43404]], ddd[:43404], batch_size=nb_batch_size,validation_data=([aaa[43404:],vector[43404:],bbb[43404:]], ddd[43404:]),epochs=nb_epoch, shuffle=True, verbose=1)
    

predictions_p = model1.predict([X1tt,ccc2,X2tt])#Evaluating the effects on Test dataset
print(np.sum(predictions_p[:,1]>0.9))
print(np.sum(predictions_p[:990,1]>0.5))
#Calculating the prediction performance and statistical results

model1.load_weights('your_model_file_name.h5')
















