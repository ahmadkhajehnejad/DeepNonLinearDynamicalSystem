exec(open('./initialize.py').read())

exec(open('./structure_definition.py').read())

def sqr_diff(X):
    tmp = K.tile(K.reshape(K.sum(K.square(X), axis=1), [-1,1]), [1,K.shape(X)[0]])
    return tmp + K.transpose(tmp) - 2*tf.matmul(X, tf.transpose(X))

kernel_sigma_2 = 1
def AE_reg_loss(x_true,w):
    return K.mean(K.exp(-sqr_diff(x_true)/kernel_sigma_2) * sqr_diff(w), axis=-1) / K.mean(K.sum(K.square(w),axis=1))

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
    return hist.history

exec(open('read_data.py').read())

IterNum_EM = 5
IterNum_CoordAsc = 5
batch_size = 1000
reg_error = []
recons_error = []
loglik = []
E_log = []
hist_0 = []
hist_1 = []
hist_2 = []

exec(open('log_tools.py').read())

log_make_dir('./tuned_params')

for iter_EM in range(IterNum_EM):
    log_make_dir('./tuned_params/' + str(iter_EM))        
    
    for i in range(len(x_all)):
        w_all[i] = enc.predict(x_all[i])
    for i in range(len(u_all)):
        v_all[i] = act_map.predict(u_all[i])        
    
    [Ezt, EztztT, Ezt_1ztT] = expectation(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0)

    if iter_EM == 0:    
        [A,b,H,C,d,Q,R,mu_0,Sig_0] = maximization(Ezt, EztztT, Ezt_1ztT, w_all, v_all, b, d)
        Rinv = np.linalg.inv(R)
        Qinv = np.linalg.inv(Q)
        
        log_update_E_log()
        log_update_loglik_recons_reg()
        

    
    for iter_CoorAsc in range(IterNum_CoordAsc):
        
        ##### update AE parameters ###########################################
        i_start, i_end = 0, -1
        for i in range(len(w_all)):
            i_end = i_start + w_all[i].shape[0]
            EzT_CT = np.matmul(Ezt[i].T, C.T)
            EzT_CT_plus_dT = EzT_CT + np.tile(d.reshape([1,-1]),[EzT_CT.shape[0],1])
            EzT_CT_Rinv_plus_dT_Rinv[i_start:i_end,:] = np.matmul(EzT_CT_plus_dT, Rinv)
            i_start = i_end
        
        Rinv_tf = tf.constant(Rinv, dtype='float32')
        
        N = x_train.shape[0]
        def LDS_loss(EzT_CT_Rinv_plus_dT_Rinv, w):
            sh = K.shape(w)
            return -tf.matmul(tf.reshape(EzT_CT_Rinv_plus_dT_Rinv,[sh[0],1,sh[1]]), tf.reshape(w,[sh[0],sh[1],1])) \
                        + 0.5 * tf.matmul(\
                                          tf.reshape(tf.matmul(w,Rinv_tf),[sh[0],1,-1])\
                                          ,tf.reshape(w,[sh[0],sh[1],1])\
                                         )
        #if (iter_EM == 0) and (iter_CoorAsc == 0):
            #AE.load_weights('./cache_0_0_simpleAE_params.h5')
        hist_0.append(AE_TRAIN(net_in=x_train, net_out=[x_train, x_train, EzT_CT_Rinv_plus_dT_Rinv], \
                        LDS_loss = LDS_loss, lr=0.005, loss_weights=[1., 1., 0.], epochs=200))
        hist_1.append(AE_TRAIN(net_in=x_train, net_out=[x_train, x_train, EzT_CT_Rinv_plus_dT_Rinv], \
                        LDS_loss = LDS_loss, lr=0.000005, loss_weights=[1., 1., 1.], epochs=1000))
            #AE.save_weights('./cache_0_0_simpleAE_params.h5')
        
        #log_update_E_log()
        
        #hist = AE_TRAIN(net_in=x_train, net_out=[x_train, x_train, EzT_CT_Rinv_plus_dT_Rinv], \
        #                LDS_loss = LDS_loss, lr=0.000005, loss_weights=[1., 0., 1.], epochs=100)
        #log_print_fit_hist(hist)
        
        #log_update_E_log()
        
        ##########  update act_map parameters #############################
        
        i_start, i_end = 0, -1
        for i in range(len(v_all)):
            i_end = i_start + v_all[i].shape[0]
            EztT_minus_Ezt_1TAT_bT = Ezt[i][:,1:].T - np.matmul(Ezt[i][:,:-1].T,A.T) - np.tile(b.T,[Ezt[i].shape[1]-1,1])
            EztT_minus_Ezt_1TAT_bT_alltimes_QinvH[i_start:i_end,:] = np.matmul(np.matmul(EztT_minus_Ezt_1TAT_bT, Qinv),H)
            i_start = i_end
            
        HTQinvH = np.matmul(np.matmul(H.T, Qinv),H)
        HTQinvH = tf.constant(HTQinvH, dtype='float32')
        def act_map_loss(EztT_minus_Ezt_1TAT_bT_alltimes_QinvH, v):
            sh = K.shape(v)
            return -tf.matmul(tf.reshape(EztT_minus_Ezt_1TAT_bT_alltimes_QinvH, [sh[0],1,sh[1]]), tf.reshape(v,[sh[0],sh[1],1]))\
                        + 0.5 * tf.matmul(tf.reshape(tf.matmul(v, HTQinvH),[sh[0],1,sh[1]]), tf.reshape(v,[sh[0],sh[1],1]))
        act_map_learning_rate = .00005
        act_map_adam = optimizers.Adam(lr=act_map_learning_rate, beta_1=0.1)
        act_map.compile(optimizer=act_map_adam, loss=act_map_loss)
        
        u_tr_len = u_train.shape[0]-np.mod(u_train.shape[0],batch_size)
        tmp_hist = act_map.fit( u_train[:u_tr_len,:] , EztT_minus_Ezt_1TAT_bT_alltimes_QinvH[:u_tr_len,:],
                  shuffle=True,
                  epochs= 200,
                  batch_size=batch_size,
                  verbose=0)
        hist_2.append(np.mean(tmp_hist.history['loss'][-10:]))
        log_print_fit_hists()
        #log_update_E_log()
        
        ############ update DLS parameters
        
        for i in range(len(x_all)):
            w_all[i] = enc.predict(x_all[i])
        for i in range(len(u_all)):
            v_all[i] = act_map.predict(u_all[i])
        
        [A,b,H,C,d,Q,R,mu_0,Sig_0] = maximization(Ezt, EztztT, Ezt_1ztT, w_all, v_all, b, d)
        Rinv = np.linalg.inv(R)
        Qinv = np.linalg.inv(Q)
        
        log_update_E_log()
        log_update_loglik_recons_reg()
        log_save_weights(iter_EM, iter_CoorAsc)        
        
    log_save_weights(iter_EM, -1)        
##################### TEST

##################################################
'''
iter_EM = 19
iter_CoorAsc = 4
AE.load_weights('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + '_AE_params.h5')
act_map.load_weights('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + '_act_map_params.h5')
[A,b,H,C,d,Q,R,mu_0,Sig_0] = pickle.load(open('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + 'LDS_params.pkl', 'rb'))
Rinv = np.linalg.inv(R)
Qinv = np.linalg.inv(Q)
[loglik,recons_error] = pickle.load(open('./results.pkl','rb'))
reg_error = pickle.load(open('./reg_error.pkl', 'rb'))
E_log = pickle.load(open('./E_log.pkl', 'rb'))
[hist_0,hist_1,hist_2] = pickle.load(open('./fit_hists.pkl', 'rb'))
'''
##########################

v_test_all = [None] * len(u_test_all)
w_test_all = [None] * len(x_test_all)
w_est_all = [None] * len(x_test_all)
x_est_all = [None] * len(x_test_all)
w_est_err_all = [None] * len(x_test_all)
x_est_err_all = [None] * len(x_test_all)

for i in range(len(x_test_all)):
    #print('i = ' + str(i))
    w_test_all[i] = enc.predict(x_test_all[i])
    v_test_all[i] = act_map.predict(u_test_all[i])
    w_est_all[i] = [None] * 25
    x_est_all[i] = [None] * 25
    w_est_err_all[i] = [None] * 25
    x_est_err_all[i] = [None] * 25
    for j in range(25):
        #print('    j = ' + str(j))
        w_est_all[i][j] = KF_predict(A,b,H,C,d,Q,R,mu_0,Sig_0,w_test_all[i][:25,:],v_test_all[i][:25+j])
        x_est_all[i][j] = dec.predict(w_est_all[i][j].reshape([j+1,-1]))
        w_est_err_all[i][j] = np.linalg.norm(w_est_all[i][j] - w_test_all[i][25:25+j+1,:], axis=1)
        x_est_err_all[i][j] = np.linalg.norm(x_est_all[i][j] - x_test_all[i][25:25+j+1,:], axis=1)
    
    #w_1_step_est[i] = [None]
    #for j in range(2,49):
    #    w_1_step_est[i][j] = KF_predict(A,b,H,C,d,Q,R,mu_0,Sig_0,w_test_all[i][:25,:],v_test_all[i][:25+j+1])

plt.figure()
plt.cla()
for i in range(20):
    plt.plot(w_est_err_all[i][24])
    
plt.figure()
plt.cla()
for i in range(20):
    plt.plot(x_est_err_all[i][24])
'''
from pykalman import KalmanFilter

kf = KalmanFilter(initial_state_mean = mu_0.reshape([-1]),
                  initial_state_covariance = Sig_0,
                  transition_matrices = A,
                  transition_offsets = b.reshape([-1]),
                  transition_covariance = Q,
                  observation_matrices = C,
                  observation_offsets = d.reshape([-1]),
                  observation_covariance = R)

'''

'''
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
'''



#######################
'''
for iter_EM in range(IterNum_EM):
    for iter_CoorAsc in range(IterNum_CoordAsc):
        AE.load_weights('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + '_AE_params.h5')
        act_map.load_weights('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + '_act_map_params.h5')
        [A,b,H,C,d,Q,R,mu_0,Sig_0] = pickle.load(open('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + 'LDS_params.pkl', 'rb'))
    
        for i in range(len(x_all)):
            w_all[i] = enc.predict(x_all[i])
        for i in range(len(u_all)):
            v_all[i] = act_map.predict(u_all[i])
    
        log_update_loglik_recons_reg()
'''