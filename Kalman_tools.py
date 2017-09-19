import numpy as np

def expectation(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0):
    
    Ezt = [None] * len(w_all)
    EztztT = [None] * len(w_all)
    Ezt_1ztT = [None] * len(w_all)
    
    for i in range(len(w_all):)
        mu_t_t = [None] * w_all.shape[0]
        Sig_t_t = [None] * w_all.shape[0]
        mu_t_t_1 = [None] * w_all.shape[0]
        Sig_t_t_1 = [None] * w_all.shape[0]
        mu_t_T = [None] * w_all.shape[0]
        Sig_t_T = [None] * w_all.shape[0]
        
        for t in range(w_all[i].shape[0]):
            
            Ezt[i] = np.zeros([mu_0.shape[0],w_all[i].shape[0]])
            EztztT[i] = [None] * w_all[i].shape[0]
            Ezt_1ztT[i] = [None] * w_all[i].shape[0]
            
            if t > 0:
                mu_t_t_1[t] = np.matmul(A,mu_t_t[t-1]) + np.matmul(H,v_all[i][t-1,:].T) + b
                Sig_t_t_1[t] = np.matmul(A, np.matmul(Sig_t_t[t-1],A.T)) + Q
            else:
                mu_t_t_1[t] = mu_0
                Sig_t_t_1[t] = Sig_0
            K = np.matmul(\
                          np.matmul(Sig_t_t_1, C.T) , \
                          np.linalg.inv(\
                                        np.matmul(\
                                                  C,
                                                  np.matmul(Sig_t_t_1[t], C.T)\
                                                 )+R\
                                       )\
                         )
            mu_t_t[t] = mu_t_t_1[t] + np.matmul(K,w_all[i][t,:].T - np.matmul(C,mu_t_t_1[t]) - d)
            Sig_t_t[t] = np.matmul(np.eye(K.shape[0]) - np.matmul(K,C), Sig_t_t_1[t])
        
            for t in reversed(range(w_all[i].shape[0])):
                if t == (w_all[i].shape[0] - 1):
                    mu_t_T[t] = mu_t_t[t]
                    Sig_t_T[t] = Sig_t_t[t]
                else:
                    L = np.matmul(Sig_t_t[t], np.matmul(A.T, np.linalg.inv(Sig_t_t_1[t+1])))
                    mu_t_T[t] = mu_t_t[t] + np.matmul(L,mu_t_T[t+1] - mu_t_t_1[t+1])
                    Sig_t_T[t] = Sig_t_t[t] + np.matmul(\
                                                        L,\
                                                        np.matmul(\
                                                                  (Sig_t_T[t+1] - Sig_t_t_1[t+1]),\
                                                                  L.T\
                                                                 )\
                                                       )
                Ezt[i][:,t] = mu_t_T[t]
                EztztT[i][t] = Sig_t_T[t] + np.matmul(mu_t_T[t], mu_t_T[t].T)
                if t < (w_all[i].shape[0] - 1):
                    Ezt_1ztT[i][t+1] = np.matmul(mu_t_t[t], mu_t_T[t+1].T) +\
                                     np.matmul(L,EztztT[i][t+1]) -\
                                     np.matmul(L,np.matmul(mu_t_t_1[t+1], mu_t_T[t+1].T))
        
    return [Ezt, EztztT, Ezt_1ztT]

def maximization(Ezt, EztztT, Ezt_1ztT,w_all, v_all):
    
    M = len(Ezt)
    T = Ezt[0].shape[1]
    
    ############################ mu_0
    
    tmp = np.zeros([Ezt[0].shape[0],1])
    for i in range(M):
        tmp = tmp + Ezt[i][:,0]
    mu_0 = tmp / M
    
    ############################ Sigma_0
    
    tmp = np.zeros(Eztzt[0].shape)
    for i in range(M):
        tmp = tmp + Eztzt[i][0]
    Sigma_0 = (tmp / M - np.matmul(mu_0, mu_0.T)
    
    ############################ H
    
    H = np.ones([Ezt[0].shape[0], v_all[0].shape[1]]) / v_all[0].shape[1]
    
    ############################ A
    
    mean_Ezt = np.zeros(Ezt[0].shape)
    mean_Ezt_1 = np.zeros(Ezt[0].shape)
    mean_Ezt_1ztT = np.zeros(Ezt_1ztT[0][0].shape)
    mean_Ezt_1zt_1T = np.zeros(EztztT[0][0].shape)
    for i in range(M):
        for t in range(1,T):
            mean_Ezt = mean_Ezt + Ezt[i][t]
            mean_Ezt_1 = mean_Ezt_1 + Ezt[i][t-1]
            mean_Ezt_1ztT = mean_Ezt_1ztT + Ezt_1ztT[i][t]
            mean_Ezt_1zt_1T = mean_Ezt_1zt_1T + EztztT[i][t-1]
    mean_Ezt = mean_Ezt / ((T-1)*M)
    mean_Ezt_1 = mean_Ezt_1 / ((T-1)*M)
    mean_Ezt_1ztT = mean_Ezt_1ztT / ((T-1)*M)
    mean_Ezt_1zt_1T = mean_Ezt_1zt_1T / ((T-1)*M)
    
    
    mean_Hvt_1 = np.zeros([Ezt[0].shape[0],1])
    mean_Hvt_1Ezt_1T = np.zeros(EztztT[0].shape)
    for i in range(M):
        for t in range(1,T):
            mean_Hvt_1 = mean_Hvt_1 + np.matmul(H,v_all[i][t-1,:].T)
            mean_Hvt_1Ezt_1T = mean_Hvt_1Ezt_1T + np.matmul(np.matmul(H,v_all[i][t-1,:].T), Ezt[i][:,t-1].T)
            
    mean_Hvt_1 = mean_Hv / ((T-1)*M)
    mean_Hvt_1Ezt_1T = mean_Hvt_1Ezt_1T / ((T-1)*M)
    
    tmp_1 = mean_Ezt_1ztT - np.matmul(mean_Ezt, mean_Ezt_1.T) + np.matmul(mean_Hvt_1, mean_Ezt_1.T)
    tmp_2 = mean_Ezt_1zt_1T - np.matmul(mean_Ezt_1, mean_Ezt_1.T) + mean_Hvt_1Ezt_1T
    
    A = np.matmul(tmp_1, np.linalg.inv(tmp_2))
    
    ############################ b
    b = mean_Ezt - mean_Ezt_1 - mean_Hvt_1
    
    ############################ Q
    tmp = np.zeros(EztztT[0][0].shape)
    for i in range(M):
        for t in range(1,T):
            tmp = tmp + EztztT[i][t] - 2*np.matmul(Ezt_1ztT[i][t].T,A.T) \
            - 2*np.matmul(np.matmul(H,v_all[i][t-1,:].T)+b,Ezt[i][:,t].T) \
            + np.matmul(A, np.matmul(EztztT[i][t-1],A.T)) \
            + 2*np.matmul(np.matmul(H,v_all[i][t-1,:].T)+b,np.matmul(Ezt[i][:,t-1].T,A.T)) \
            + np.matmul(np.matmul(H,v_all[i][t-1,:].T)+b, (np.matmul(H,v_all[i][t-1,:].T)+b).T)
    Q = tmp / ((T-1)*M)
    
    ############################ C
    mean_wt = np.zeros([w_all[0].shape[1],1])
    mean_wtEztT = np.zeros([w_all[0].shape[1], Ezt[0].shape[0]])
    mean_Ezt = np.zeros([Ezt[0].shape[0], 1])
    for i in range(M):
        for t in range(T):
            mean_wt = mean_wt + w_all[i][t,:].T
            mean_wtEztT = mean_wtEztT + np.matmul(w_all[i][t,:].T,Ezt[i][:,t].T)
            mean_Ezt = mean_Ezt + Ezt[i][:,t]
    mean_wt = mean_wt / (T*M)
    mean_wtEztT = mean_wtEztT / (T*M)
    mean_Ezt = mean_Ezt / (T*M)
    
    tmp_1 = mean_wtEztT - np.matmul(mean_wt, mean_Ezt.T)
    tmp_2 = np.linalg.inv(np.matmul(mean_Ezt, mean_Ezt.T))
    C = np.matmul(tmp_1, tmp_2.T)
    ############################ d
    
    d = mean_wt - np.matmul(C,mean_Ezt)
    
    ############################ R
    mean_wt_dwt_dT = np.zeros([d.shape[0], d.shape[0]])
    mean_EztEztT = np.zeros(EztztT[0].shape)
    mean_wt_dEztT = np.zeros([d.shape[0],Ezt[0].shape[0]])
    for i in range(M):
        for t in range(T):
            mean_wt_dwt_dT = mean_wt_dwt_dT + np.matmul(w_all[i][t,:].T - d, (w_all[i][t,:].T - d).T)
            mean_EztEztT = mean_EztEztT + EztztT[i][t]
            mean_wt_dEztT = mean_wt_dEztT + np.matmul(w_all[i][t,:].T - d,Ezt[i][:,t].T)
    mean_wt_dwt_dT = mean_wt_dwt_dT / (T*M)
    mean_EztEztT = mean_EztEztT / (T*M)
    mean_wt_dEztT = mean_wt_dEztT / (T*M)
    
    R = mean_wt_dwt_dT + np.matmul(C, np.matmul(mean_EztEztT.T,C.T)) \
        - 2*np.matmul(mean_wt_dEztT,np.matmul)
    return [A,b,H,C,d,Q,R,mu_0,Sig_0]



def EM_step(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0):
    [Ezt, EztztT, Ezt_1ztT] = expectation(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0)
    [A,b,H,C,d,Q,R,mu_0,Sig_0] = maximization(Ezt, EztztT, Ezt_1ztT, w_all, v_all)
    return [A,b,H,C,d,Q,R,mu_0,Sig_0]