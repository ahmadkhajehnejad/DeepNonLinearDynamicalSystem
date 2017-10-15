import numpy as np
import pickle
import keras
from keras.models import Model , Sequential
from keras.layers import Dense, Input, Reshape, Lambda, Concatenate
from keras import backend as K
import tensorflow as tf
from keras import objectives , optimizers, callbacks
import matplotlib.pyplot as plt
import h5py
import os
import Kalman_tools
from Kalman_tools import expectation, maximization, EM_step

w_dim, z_dim, v_dim, x_dim, u_dim = 2, 2, 2, 40*40, 2
    
mu_0, Sig_0 = np.zeros([z_dim,1]), np.eye(z_dim)
A,b,H,Q = np.eye(z_dim) + np.random.uniform(-0.1,0.1,z_dim*z_dim).reshape([z_dim,z_dim]), np.zeros([z_dim,1]), np.ones([z_dim, v_dim])/v_dim, np.eye(z_dim)
C,d,R = np.ones([w_dim, z_dim])/z_dim + np.random.uniform(-0.1,0.1,w_dim*z_dim).reshape([w_dim,z_dim]), np.zeros([w_dim,1]), np.eye(w_dim)

x = Input(shape=(x_dim,))
h = Dense(256, activation='relu')(x)
w_mean = Dense(w_dim)(h)
w_log_var = Dense(w_dim)(h)

def sampling(args):
    w_mean, w_log_var = args
    epsilon = K.random_normal(shape=(K.shape(w_mean)[0], w_dim), mean=0.,
                              stddev=1.)
    return w_mean + K.exp(w_log_var / 2) * epsilon

w = Lambda(sampling, output_shape=(w_dim,))([w_mean, w_log_var])

enc = Model([x],[w])

decoder_h = Dense(256, activation='relu')
decoder_mean = Dense(x_dim, activation='sigmoid')
h_decoded = decoder_h(w)
x_bar = decoder_mean(h_decoded)

decoder_input = Input(shape=(w_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
dec = Model(decoder_input, _x_decoded_mean)

AE = Model([x],[x_bar,x_bar,w])

u_in = Input(shape=(u_dim,))
h_u = Dense(5,activation='relu')(u_in)
u_out = Dense(v_dim, activation='sigmoid')(h_u)

act_map = Model(u_in,u_out)

def AE_recons_loss(x_true, x_bar):
    return x_dim * keras.metrics.binary_crossentropy(x_true, x_bar)#keras.losses.mean_squared_error(x_true, x_bar) ## might be better to be changed to binary_cross_entropy
    
def AE_KL_loss(true_out, est_out):
    return - 0.5 * K.sum(1 + w_log_var - K.square(w_mean) - K.exp(w_log_var), axis=-1)

def AE_TRAIN(net_in, net_out, LDS_loss, lr, loss_weights, epochs):
    AE_adam = optimizers.Adam(lr=lr, beta_1=0.1)
    AE.compile(optimizer=AE_adam, loss=[AE_recons_loss, AE_KL_loss, LDS_loss], \
                      loss_weights=loss_weights)
    hist = AE.fit( net_in, net_out,
              shuffle=True,
              epochs= epochs,
              batch_size=batch_size,
              verbose=1)
    return hist

print('data load start')
[tmp_X, tmp_U, _] = pickle.load(open('plane_random_trajectory_train', 'rb'))
print('data load finish')

x_all = [None] * len(tmp_X)
u_all = [None] * len(tmp_U)
for i in range(len(tmp_X)):
    x_all[i] = tmp_X[i].reshape([tmp_X[i].shape[0],-1])
    u_all[i] = tmp_U[i][:-1,:].reshape([tmp_U[i].shape[0]-1,-1])

n_train = 0;
for i in range(len(x_all)):
    n_train = n_train + x_all[i].shape[0]

x_train = np.zeros([n_train,x_dim])
i_start, i_finish = 0, -1
for i in range(len(x_all)):
    i_finish = i_start + len(x_all[i])
    x_train[i_start:i_finish,:] = x_all[i]
    i_start = i_finish
'''
x_train = x_all[0]
for i in range(1,len(x_all)):
    x_train = np.concatenate([x_train, x_all[i]])
'''

print('train data is ready.')

print('data load start')
[tmp_X, _, _] = pickle.load(open('plane_random_trajectory_test', 'rb'))
print('data load finish')

x_test_all = [None] * len(tmp_X)
for i in range(len(tmp_X)):
    x_test_all[i] = tmp_X[i].reshape([tmp_X[i].shape[0],-1])

x_test = x_test_all[0]
for i in range(1,len(x_test_all)):
    x_test = np.concatenate([x_test, x_test_all[i]])

print('test data is ready.')

EzT_CT_Rinv_minus_dT_Rinv = np.zeros([x_train.shape[0],w_dim])

u_train = u_all[0]
for i in range(1,len(u_all)):
    u_train = np.concatenate([u_train, u_all[i]])
EztT_minus_Ezt_1TAT_bT_alltimes_QinvH = np.zeros([u_train.shape[0],v_dim])

w_all = [None] * len(x_all)
v_all = [None] * len(u_all)

IterNum_EM = 50
IterNum_CoordAsc = 5
recons_error = []
#IterNum_DeepTrain = 1000
batch_size = 100

loglik = []
for iter_EM in range(IterNum_EM):
    
    if not(os.path.isdir('./tuned_params')):
        os.mkdir('./tuned_params')
    
    if not(os.path.isdir('./tuned_params/' + str(iter_EM))):
        os.mkdir('./tuned_params/' + str(iter_EM))
    
    for i in range(len(x_all)):
        w_all[i] = enc.predict(x_all[i])
    for i in range(len(u_all)):
        v_all[i] = act_map.predict(u_all[i])

    if iter_EM == 0:
        loglik.append(Kalman_tools.log_likelihood(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0))
        print('')
        print('loglik = ')
        print(loglik)
        [x_bar, _, _] = AE.predict(x_test)
        tmp = np.mean((x_bar - x_test) ** 2)
        recons_error.append(tmp)
        print('recons_error = ')
        print(recons_error)

    [Ezt, EztztT, Ezt_1ztT] = expectation(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0)
    
    for iter_CoorAsc in range(IterNum_CoordAsc):
        
        for i in range(len(x_all)):
            w_all[i] = enc.predict(x_all[i])
        for i in range(len(u_all)):
            v_all[i] = act_map.predict(u_all[i])
        
        [A,b,H,C,d,Q,R,mu_0,Sig_0] = maximization(Ezt, EztztT, Ezt_1ztT, w_all, v_all, b, d)
        Rinv = np.linalg.inv(R)
        Qinv = np.linalg.inv(Q)
        
        i_start, i_end = 0, -1
        for i in range(len(w_all)):
            i_end = i_start + w_all[i].shape[0]
            EzT_CT = np.matmul(Ezt[i].T, C.T)
            EzT_CT_minus_dT = EzT_CT - np.tile(d.reshape([1,-1]),[EzT_CT.shape[0],1])
            EzT_CT_Rinv_minus_dT_Rinv[i_start:i_end,:] = np.matmul(EzT_CT_minus_dT, Rinv)
            i_start = i_end
        
        Rinv = tf.constant(Rinv, dtype='float32')
        
        N = x_train.shape[0]
        def LDS_loss(EzT_CT_Rinv_minus_dT_Rinv, w):
            sh = K.shape(w)
            return -tf.matmul(tf.reshape(EzT_CT_Rinv_minus_dT_Rinv,[sh[0],1,sh[1]]), tf.reshape(w,[sh[0],sh[1],1])) \
                        + 0.5 * tf.matmul(\
                                          tf.reshape(tf.matmul(w,Rinv),[sh[0],1,-1])\
                                          ,tf.reshape(w,[sh[0],sh[1],1])\
                                         )
        if (iter_EM == 0) and (iter_CoorAsc == 0):
            AE.load_weights('./cache_0_0_variationalAE_params.h5')
            #hist = AE_TRAIN(net_in=x_train, net_out=[x_train, np.zeros([N,w_dim]), EzT_CT_Rinv_minus_dT_Rinv], LDS_loss = LDS_loss, lr=0.001, loss_weights=[1., 1., 0.], epochs=200)
            #print('-------------------')
            #print(np.mean(hist.history[list(hist.history.keys())[1]][-10:]))
            #print(np.mean(hist.history[list(hist.history.keys())[3]][-10:]))
            #print('-------------------')
            #AE.save_weights('./cache_0_0_variationalAE_params.h5')

        
        hist = AE_TRAIN(net_in=x_train, net_out=[x_train, np.zeros([N,w_dim]), EzT_CT_Rinv_minus_dT_Rinv], LDS_loss = LDS_loss, lr=0.001, loss_weights=[1., 1., 1.], epochs=50)
        print('-------------------')
        print(np.mean(hist.history[list(hist.history.keys())[1]][-10:]))
        print(np.mean(hist.history[list(hist.history.keys())[3]][-10:]))
        print('-------------------')
        
        hist = AE_TRAIN(net_in=x_train, net_out=[x_train, np.zeros([N,w_dim]), EzT_CT_Rinv_minus_dT_Rinv], LDS_loss = LDS_loss, lr=0.0001, loss_weights=[1., 1., .00001], epochs=50)
        print('-------------------')
        print(np.mean(hist.history[list(hist.history.keys())[1]][-10:]))
        print(np.mean(hist.history[list(hist.history.keys())[3]][-10:]))
        print('-------------------')
        
        hist = AE_TRAIN(net_in=x_train, net_out=[x_train, np.zeros([N,w_dim]), EzT_CT_Rinv_minus_dT_Rinv], LDS_loss = LDS_loss, lr=0.00001, loss_weights=[1., 1., .00001], epochs=50)
        print('-------------------')
        print(np.mean(hist.history[list(hist.history.keys())[1]][-10:]))
        print(np.mean(hist.history[list(hist.history.keys())[3]][-10:]))
        print('-------------------')
        
        i_start, i_end = 0, -1
        for i in range(len(v_all)):
            i_end = i_start + v_all[i].shape[0]
            EztT_minus_Ezt_1TAT_bT = Ezt[i][:,1:].T - np.matmul(Ezt[i][:,:-1].T,A.T) - np.tile(b.T,[Ezt[i].shape[1]-1,1])
            EztT_minus_Ezt_1TAT_bT_alltimes_QinvH[i_start:i_end,:] = np.matmul(np.matmul(EztT_minus_Ezt_1TAT_bT, Qinv),H)
            
        HTQinvH = np.matmul(np.matmul(H.T, Qinv),H)
        HTQinvH = tf.constant(HTQinvH, dtype='float32')
        def act_map_loss(EztT_minus_Ezt_1TAT_bT_alltimes_QinvH, v):
            sh = K.shape(v)
            return -tf.matmul(tf.reshape(EztT_minus_Ezt_1TAT_bT_alltimes_QinvH, [sh[0],1,sh[1]]), tf.reshape(v,[sh[0],sh[1],1]))\
                        + 0.5 * tf.matmul(tf.reshape(tf.matmul(v, HTQinvH),[sh[0],1,sh[1]]), tf.reshape(v,[sh[0],sh[1],1]))
        act_map_learning_rate = .0005
        act_map_adam = optimizers.Adam(lr=act_map_learning_rate, beta_1=0.1)
        act_map.compile(optimizer=act_map_adam, loss=act_map_loss)
        
        u_tr_len = u_train.shape[0]-np.mod(u_train.shape[0],batch_size)
        hist = act_map.fit( u_train[:u_tr_len,:] , EztT_minus_Ezt_1TAT_bT_alltimes_QinvH[:u_tr_len,:],
                  shuffle=True,
                  epochs= 100,
                  batch_size=batch_size,
                  verbose=0)
        print(np.mean(hist.history['loss'][-10:]))
        
        loglik.append(Kalman_tools.log_likelihood(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0))
        print('')
        print('loglik = ')
        print(loglik)
        [x_bar, _, _] = AE.predict(x_test)
        tmp = np.mean((x_bar - x_test) ** 2)
        recons_error.append(tmp)
        print('recons_error = ')
        print(recons_error)

        
        AE.save_weights('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + '_AE_params.h5')
        act_map.save_weights('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + '_act_map_params.h5')
        pickle.dump([A,b,H,C,d,Q,R,mu_0,Sig_0], open('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + 'LDS_params.pkl', 'wb'))
        pickle.dump([loglik,recons_error], open('./results.pkl','wb'))
        
    
    AE.save_weights('./tuned_params/' + str(iter_EM) + '_AE_params.h5')
    act_map.save_weights('./tuned_params/' + str(iter_EM) + '_act_map_params.h5')
    pickle.dump([A,b,H,C,d,Q,R,mu_0,Sig_0], open('./tuned_params/' + str(iter_EM) + '_LDS_params.pkl', 'wb'))
    pickle.dump(loglik, open('./loglikelihood.pkl', 'wb'))
    
    
##################### TEST

AE.load_weights('./tuned_params/2/0_AE_params.h5')
act_map.load_weights('./tuned_params/0/0_act_map_params.h5')
[A,b,H,C,d,Q,R,mu_0,Sig_0] = pickle.load(open('./tuned_params/2/0LDS_params.pkl','rb'))

from pykalman import KalmanFilter

kf = KalmanFilter(initial_state_mean = mu_0.reshape([-1]),
                  initial_state_covariance = Sig_0,
                  transition_matrices = A,
                  transition_offsets = b.reshape([-1]),
                  transition_covariance = Q,
                  observation_matrices = C,
                  observation_offsets = d.reshape([-1]),
                  observation_covariance = R)

[x_test, u_test, _] = pickle.load(open('plane_random_trajectory_test', 'rb'))
x_test = x_test.reshape([x_test.shape[0],-1])
u_test = u_test[:-1,:]

w_test = enc.predict(x_test)
[z_est, z_est_var] = kf.filter(w_test)
w_est = np.matmul(z_est, C.T) + np.tile(d.reshape([1,-1]),[z_est.shape[0],1])
x_est = dec.predict(w_est)


ii = 90
plt.figure()
plt.subplot(2,1,1)            
plt.imshow(x_test[ii].reshape(40,40), cmap='Greys')
plt.subplot(2,1,2)
plt.imshow(x_est[ii].reshape(40,40), cmap='Greys')

#################


[x_bar, _, _] = AE.predict(x_test)
np.mean((x_bar - x_test) ** 2)

ii = 40
plt.figure()
plt.subplot(2,1,1)            
plt.imshow(x_test[ii].reshape(40,40), cmap='Greys')
plt.subplot(2,1,2)
plt.imshow(x_bar[ii].reshape(40,40), cmap='Greys')

#################

[x_bar, _, _] = AE.predict(x_train)
np.mean((x_bar - x_train) ** 2)

ii = 40
plt.figure()
plt.subplot(2,1,1)            
plt.imshow(x_train[ii].reshape(40,40), cmap='Greys')
plt.subplot(2,1,2)
plt.imshow(x_bar[ii].reshape(40,40), cmap='Greys')


#################
def nearest_w(w, w_train):
    mn, mn_i = -1, -1
    for i in range(w_train.shape[0]):
        d = np.linalg.norm(w - w_train[i,:])
        if (i == 0) or d < mn:
            mn = d
            mn_i = i
    return w_train[mn_i,:]

w_test = enc.predict(x_test)
w_train = enc.predict(x_train)
w_0 = w_test[10]
w_1 = w_test[70]
delta_w = (w_1 - w_0) / 9
plt.figure()
plt.subplot(2,6,1)
plt.imshow(x_test[10].reshape(40,40), cmap='Greys')
for i in range(10):
    i
    #w_t = nearest_w(w_0 + i*delta_w, w_train)
    w_t = w_0 + i*delta_w
    x_t = dec.predict(w_t.reshape([1,-1]))
    plt.subplot(2,6,i+2)
    plt.imshow(x_t.reshape(40,40), cmap='Greys')
plt.subplot(2,6,12)
plt.imshow(x_test[70].reshape(40,40), cmap='Greys')
