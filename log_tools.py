def log_make_dir(dirname):
    if not(os.path.isdir(dirname)):
        os.mkdir(dirname)
    
def log_update_loglik_recons():
    loglik.append(Kalman_tools.log_likelihood(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0))
    print('')
    print('loglik = ')
    print(loglik)
    [x_bar, _, _] = AE.predict(x_test)
    tmp = np.mean((x_bar - x_test) ** 2)
    recons_error.append(tmp)
    print('recons_error = ')
    print(recons_error)

def log_save_weights(iter_EM, iter_CoorAsc):
    if iter_CoorAsc == -1:
        AE.save_weights('./tuned_params/' + str(iter_EM) + '_AE_params.h5')
        act_map.save_weights('./tuned_params/' + str(iter_EM) + '_act_map_params.h5')
        pickle.dump([A,b,H,C,d,Q,R,mu_0,Sig_0], open('./tuned_params/' + str(iter_EM) + '_LDS_params.pkl', 'wb'))
        pickle.dump(loglik, open('./loglikelihood.pkl', 'wb'))
    else:
        AE.save_weights('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + '_AE_params.h5')
        act_map.save_weights('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + '_act_map_params.h5')
        pickle.dump([A,b,H,C,d,Q,R,mu_0,Sig_0], open('./tuned_params/' + str(iter_EM) + '/' + str(iter_CoorAsc) + 'LDS_params.pkl', 'wb'))
        pickle.dump([loglik,recons_error], open('./results.pkl','wb'))
        
def log_print_fit_hist(hist):
    print('-------------------')
    print(np.mean(hist.history[list(hist.history.keys())[1]][-10:]))
    print(np.mean(hist.history[list(hist.history.keys())[2]][-10:]))
    print(np.mean(hist.history[list(hist.history.keys())[3]][-10:]))
    print('-------------------')
    
def log_print_E():
    E = E_log_P_x_and_z(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0)
    print('E[log...] = ' E)