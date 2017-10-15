exec(open('./initialize.py').read())
exec(open('./structure_definition.py').read())

def sqr_diff(X):
    tmp = K.tile(K.reshape(K.sum(K.square(X), axis=1), [-1,1]), [1,K.shape(X)[0]])
    return tmp + K.transpose(tmp) - 2*tf.matmul(X, tf.transpose(X))

kernel_sigma_2 = 1
def AE_reg_loss(x_true,w):
    return K.mean(K.exp(-sqr_diff(x_true)/kernel_sigma_2) * sqr_diff(w), axis=-1)

def AE_recons_loss(x_true, x_bar):
    recon = x_dim * keras.losses.mean_squared_error(x_true, x_bar) #keras.metrics.binary_crossentropy(x_true, x_bar)# might be better to be changed to binary_cross_entropy
    return recon

def AE_TRAIN(net_in, net_out, LDS_loss, lr, loss_weights, epochs):
    AE_adam = optimizers.Adam(lr=lr, beta_1=0.1)
    AE.compile(optimizer=AE_adam, loss=[AE_recons_loss, AE_reg_loss, LDS_loss], \
                      loss_weights=loss_weights)
    hist = AE.fit( net_in, net_out,
              shuffle=True,
              epochs= epochs,
              batch_size=batch_size,
              verbose=0)
    return hist

exec(open('read_data.py').read())

IterNum_EM = 50
IterNum_CoordAsc = 5
batch_size = 100
recons_error = []
loglik = []

exec(open('log_tools.py').read())

log_make_dir('./tuned_params')

for iter_EM in range(IterNum_EM):
    log_make_dir('./tuned_params/' + str(iter_EM))        
    
    for i in range(len(x_all)):
        w_all[i] = enc.predict(x_all[i])
    for i in range(len(u_all)):
        v_all[i] = act_map.predict(u_all[i])
    
    if iter_EM == 0:
        log_update_loglik_recons()
    
    [Ezt, EztztT, Ezt_1ztT] = expectation(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0)
    
    for iter_CoorAsc in range(IterNum_CoordAsc):
        
        for i in range(len(x_all)):
            w_all[i] = enc.predict(x_all[i])
        for i in range(len(u_all)):
            v_all[i] = act_map.predict(u_all[i])
        
        [A,b,H,C,d,Q,R,mu_0,Sig_0] = maximization(Ezt, EztztT, Ezt_1ztT, w_all, v_all, b, d)
        Rinv = np.linalg.inv(R)
        Qinv = np.linalg.inv(Q)
        
        log_print_E()
        
        i_start, i_end = 0, -1
        for i in range(len(w_all)):
            i_end = i_start + w_all[i].shape[0]
            EzT_CT = np.matmul(Ezt[i].T, C.T)
            EzT_CT_minus_dT = EzT_CT - np.tile(d.reshape([1,-1]),[EzT_CT.shape[0],1])
            EzT_CT_Rinv_minus_dT_Rinv[i_start:i_end,:] = np.matmul(EzT_CT_minus_dT, Rinv)
            i_start = i_end
        
        Rinv_tf = tf.constant(Rinv, dtype='float32')
        
        N = x_train.shape[0]
        def LDS_loss(EzT_CT_Rinv_minus_dT_Rinv, w):
            sh = K.shape(w)
            return -tf.matmul(tf.reshape(EzT_CT_Rinv_minus_dT_Rinv,[sh[0],1,sh[1]]), tf.reshape(w,[sh[0],sh[1],1])) \
                        + 0.5 * tf.matmul(\
                                          tf.reshape(tf.matmul(w,Rinv_tf),[sh[0],1,-1])\
                                          ,tf.reshape(w,[sh[0],sh[1],1])\
                                         )
        #if (iter_EM == 0) and (iter_CoorAsc == 0):
            #AE.load_weights('./cache_0_0_simpleAE_params.h5')
        hist = AE_TRAIN(net_in=x_train, net_out=[x_train, x_train, EzT_CT_Rinv_minus_dT_Rinv], LDS_loss = LDS_loss, lr=0.001, loss_weights=[1., 1., .001], epochs=100)
        log_print_fit_hist(hist)
            #AE.save_weights('./cache_0_0_simpleAE_params.h5')
        
        hist = AE_TRAIN(net_in=x_train, net_out=[x_train, x_train, EzT_CT_Rinv_minus_dT_Rinv], LDS_loss = LDS_loss, lr=0.0003, loss_weights=[1., 1., .001], epochs=100)
        log_print_fit_hist(hist)
        
        log_print_E()
        
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
        log_print_E()
        
        log_update_loglik_recons()
        log_save_weights(iter_EM, iter_CoorAsc)        
        
    log_save_weights(iter_EM, -1)        
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


[x_bar,_] = AE.predict(x_test)
np.mean((x_bar - x_test) ** 2)

ii = 40
plt.figure()
plt.subplot(2,1,1)            
plt.imshow(x_test[ii].reshape(40,40), cmap='Greys')
plt.subplot(2,1,2)
plt.imshow(x_bar[ii].reshape(40,40), cmap='Greys')

#################

[x_bar,_] = AE.predict(x_train)
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

plt.figure()
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


[s_gr, x_gr] = pickle.load(open('all_states', 'rb'))
x_gr = x_gr.reshape([x_gr.shape[0],-1])
w_gr = enc.predict(x_gr)
plt.figure(figsize=(6, 6))
plt.scatter(w_gr[:,0], w_gr[:, 1], c= np.arange(1296),linewidth = 0)
       
#plt.figure(figsize=(6, 6))
#plt.scatter(s_gr[:,0], s_gr[:, 1], c= np.arange(1296),linewidth = 0)
