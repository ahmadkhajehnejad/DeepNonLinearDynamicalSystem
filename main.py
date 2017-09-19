import numpy as np
import pickle
from keras.models import Model , Sequential
from keras.layers import Dense, Input, Reshape, Lambda, Concatenate
from keras import backend as K
import tensorflow as tf
from keras import objectives , optimizers, callbacks
import matplotlib.pyplot as plt

from Kalman_tools import expectation, maximization, EM_step

A,b,C,d,Q,R,mu_0,Sig_0 = np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([])
Rinv = np.asarrya([])

enc = Sequential()
'''
Definition of enc
'''

dec = Sequential()
'''
Definition of dec
'''

x = Input(shape=(input_size,))
w = enc(x)
x_bar = dec(x)
model = Model([x],[x_bar,w])

def model_loss_1(x_true, x_bar):
    return keras.losses.mean_error(x_true, x_bar) ## might be better to be changed to binary_cross_entropy

def model_loss_2(EzT_CT_Rinv_minus_dT_Rinv, w):
    sh = K.shape(w)
    return tf.matmul(tf.reshape(w,[sh[0],1,sh[1]]), tf.reshape(EzT_CT_Rinv_minus_dT_Rinv,[sh[0],sh[1],1])) \
                - 0.5 * tf.matmul(\
                                  tf.reshape(tf.matmul(w,Rinv),[sh[0],1,-1])\
                                  ,tf.reshape(w,[sh[0],sh[1],1])\
                                 )

learning_rate =   0.0005 
my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.1)
model.compile(optimizer=my_adam, loss=[model_loss_1, model_loss_2], \
                  loss_weights=[1.,1.])

x_train = np.asanyarray([])
for i in range(len(x_all)):
    x_train = np.concatenate(x_train, x_all[i])

EzT_CT_Rinv_minus_dT_Rinv = np.zeros([x_train.shape[0], latent_dim])
w_all = [None] * len(x_all)

IterNum_EM = 50
IterNum_CoordAsc = 50
IterNum_DeepTrain = 100

for iter_EM in range(IterNum_EM):
    for i in len(x_all):
        w_all[i] = enc.predict(x_all[i])
    [Ezt, EztztT, Ezt_1ztT] = expectation(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0)
    
    i_start, i_end = 0, -1
    for i in range(len(w_all)):
        i_end = i_start + w_all[i].shape[0]
        EzT_CT = np.matmul(Ezt[i].T, C.T)
        EzT_CT_minus_dT = EzT_CT - np.tile(d.reshape([1,-1]),[EzT_CT.shape[0],1])
        EzT_CT_Rinv_minus_dT_Rinv[i_start:i_end,:] = np.matmul(EzT_CT_minus_dT, Rinv
        i_start = i_end
        
    
    for iter_CoorAsc in range(IterNum_CoordAsc)
        for i in len(x_all):
            w_all[i] = enc.predict(x_all[i])
        [A,b,H,C,d,Q,R,mu_0,Sig_0,EzT] = maximization(w_all,Ezt, EztztT, Ezt_1ztT)
        Rinv = np.linalg.inv(R)
        Qinv = np.linalg.inv(Q)
        
        i_start, i_end = 0, -1
        for i in range(len(w_all)):
            i_end = i_start + w_all[i].shape[0]
            EzT_CT = np.matmul(Ezt[i].T, C.T)
            EzT_CT_minus_dT = EzT_CT - np.tile(d.reshape([1,-1]),[EzT_CT.shape[0],1])
            EzT_CT_Rinv_minus_dT_Rinv[i_start:i_end,:] = np.matmul(EzT_CT_minus_dT, Rinv)
            i_start = i_end
            
        model.fit( x_train , [x_train , EzT_CT_Rinv_minus_dT_Rinv],
                  shuffle=True,
                  nb_epoch= IterNum_DeepTrain,
                  batch_size=batch_size)
