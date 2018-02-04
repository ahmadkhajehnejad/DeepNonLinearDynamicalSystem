def log_make_dir(dirname):
    if not(os.path.isdir(dirname)):
        os.mkdir(dirname)
    
    
def _reg_loss(x,w):
    return np.mean(K.eval(AE_reg_loss(K.constant(x), K.constant(w))))
    
def log_update_loglik_recons_reg():
    loglik.append(Kalman_tools.log_likelihood(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0))
    print('')
    print('loglik = ')
    print(loglik)
    [x_bar, w, _] = AE.predict(x_test)
    tmp = np.mean((x_bar - x_test) ** 2)
    recons_error.append(tmp)
    print('recons_error = ')
    print(recons_error)
    reg_error.append(_reg_loss(x_test,w))
    print('reg_error = ')
    print(reg_error)

def log_save_weights(iter_EM, iter_CoorAsc):
    if iter_CoorAsc == -1:
        AE.save_weights('./tuned_params/' + str(iter_EM) + '_AE_params.h5')
        act_map.save_weights('./tuned_params/' + str(iter_EM) + '_act_map_params.h5')
        pickle.dump([A,b,H,C,d,Q,R,mu_0,Sig_0], open('./tuned_params/' + str(iter_EM) + '_LDS_params.pkl', 'wb'))
        pickle.dump([loglik,recons_error], open('./loglikelihood.pkl', 'wb'))
        pickle.dump(E_log, open('./E_log.pkl', 'wb'))
        pickle.dump([hist_0,hist_1,hist_2], open('./fit_hists.pkl', 'wb'))
    else:
        AE.save_weights('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + '_AE_params.h5')
        act_map.save_weights('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + '_act_map_params.h5')
        pickle.dump([A,b,H,C,d,Q,R,mu_0,Sig_0], open('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + 'LDS_params.pkl', 'wb'))
        pickle.dump([loglik,recons_error], open('./results.pkl','wb'))
        pickle.dump(E_log, open('./E_log.pkl', 'wb'))
        pickle.dump([hist_0,hist_1,hist_2], open('./fit_hists.pkl', 'wb'))

        
def log_print_fit_hists():
    print('-------------------')
    print(np.mean(hist_0[-1][list(hist_0[-1].keys())[1]][-10:]))
    print(np.mean(hist_0[-1][list(hist_0[-1].keys())[2]][-10:]))
    print(np.mean(hist_0[-1][list(hist_0[-1].keys())[3]][-10:]))
    print('-------------------')
    print('-------------------')
    print(np.mean(hist_1[-1][list(hist_1[-1].keys())[1]][-10:]))
    print(np.mean(hist_1[-1][list(hist_1[-1].keys())[2]][-10:]))
    print(np.mean(hist_1[-1][list(hist_1[-1].keys())[3]][-10:]))
    print('-------------------')
    print('-------------------')
    print(hist_2[-1])
    print('-------------------')
    
def log_update_E_log():
    for i in range(len(x_all)):
        w_all[i] = enc.predict(x_all[i])
    for i in range(len(u_all)):
        v_all[i] = act_map.predict(u_all[i])
    E_log.append(E_log_P_x_and_z(Ezt, EztztT, Ezt_1ztT,w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0))
    print('E[log...] = ', E_log)