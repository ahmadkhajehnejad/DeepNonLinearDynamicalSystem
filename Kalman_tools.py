import numpy as np
from pykalman import KalmanFilter
from scipy.stats import multivariate_normal

def expectation(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0):
    
    M = len(w_all)
    T = w_all[0].shape[0]
    
    Ezt = [None] * M
    EztztT = [None] * M
    Ezt_1ztT = [None] * M
    
    z_dim = mu_0.shape[0]
    for i in range(M):
       
        Ezt[i] = np.zeros([z_dim,T])
        EztztT[i] = [None] * T
        Ezt_1ztT[i] = [None] * T
            
        
        mu_t_t = [None] * T
        Sig_t_t = [None] * T
        mu_t_t_1 = [None] * T
        Sig_t_t_1 = [None] * T
        mu_t_T = [None] * T
        Sig_t_T = [None] * T
        
        for t in range(T):
            if t > 0:
                mu_t_t_1[t] = np.matmul(A,mu_t_t[t-1]) + np.matmul(H,v_all[i][t-1,:].reshape([-1,1])) + b
                Sig_t_t_1[t] = np.matmul(A, np.matmul(Sig_t_t[t-1],A.T)) + Q
            else:
                mu_t_t_1[t] = mu_0
                Sig_t_t_1[t] = Sig_0
            K = np.matmul(\
                          np.matmul(Sig_t_t_1[t], C.T) , \
                          np.linalg.pinv(\
                                        np.matmul(\
                                                  C,
                                                  np.matmul(Sig_t_t_1[t], C.T)\
                                                 )+R\
                                       )\
                         )
            mu_t_t[t] = mu_t_t_1[t] + np.matmul(K,w_all[i][t,:].reshape([-1,1]) - np.matmul(C,mu_t_t_1[t]) - d)
            Sig_t_t[t] = np.matmul(np.eye(K.shape[0]) - np.matmul(K,C), Sig_t_t_1[t])
                
        for t in reversed(range(T)):
            if t == (T - 1):
                mu_t_T[t] = mu_t_t[t]
                Sig_t_T[t] = Sig_t_t[t]
            else:
                L = np.matmul(Sig_t_t[t], np.matmul(A.T, np.linalg.pinv(Sig_t_t_1[t+1])))
                mu_t_T[t] = mu_t_t[t] + np.matmul(L,mu_t_T[t+1] - mu_t_t_1[t+1])
                Sig_t_T[t] = Sig_t_t[t] + np.matmul(\
                                                    L,\
                                                    np.matmul(\
                                                              (Sig_t_T[t+1] - Sig_t_t_1[t+1]),\
                                                              L.T\
                                                             )\
                                                   )
            Ezt[i][:,t] = mu_t_T[t].reshape([-1])
            EztztT[i][t] = Sig_t_T[t] + np.matmul(mu_t_T[t], mu_t_T[t].T)
            if t < (T - 1):
                Ezt_1ztT[i][t+1] = np.matmul(mu_t_t[t], mu_t_T[t+1].T) +\
                                 np.matmul(L,EztztT[i][t+1]) -\
                                 np.matmul(L,np.matmul(mu_t_t_1[t+1], mu_t_T[t+1].T))
    
    return [Ezt, EztztT, Ezt_1ztT]

def maximization(Ezt, EztztT, Ezt_1ztT,w_all, v_all):
    
    w_dim = w_all[0].shape[1]
    v_dim = v_all[0].shape[1]
    z_dim = Ezt[0].shape[0]
    
    M = len(Ezt)
    T = Ezt[0].shape[1]
    
    ############################ mu_0
    
    tmp = np.zeros([z_dim,1])
    for i in range(M):
        tmp = tmp + Ezt[i][:,0].reshape([-1,1])
    mu_0 = tmp / M
    
    ############################ Sigma_0
    
    tmp = np.zeros([z_dim,z_dim])
    for i in range(M):
        tmp = tmp + EztztT[i][0]
    Sig_0 = (tmp / M) - np.matmul(mu_0, mu_0.T)
    
    ############################ H
    
    H = np.ones([z_dim, v_dim]) / v_dim
    
    ############################ A
    
    mean_Ezt = np.zeros([z_dim,1])
    mean_Ezt_1 = np.zeros([z_dim,1])
    mean_Ezt_1ztT = np.zeros([z_dim,z_dim])
    mean_Ezt_1zt_1T = np.zeros([z_dim,z_dim])
    mean_Hvt_1 = np.zeros([z_dim,1])
    mean_Hvt_1Ezt_1T = np.zeros([z_dim,z_dim])
    
    for i in range(M):
        for t in range(1,T):
            mean_Ezt = mean_Ezt + Ezt[i][:,t].reshape([-1,1])
            mean_Ezt_1 = mean_Ezt_1 + Ezt[i][:,t-1].reshape([-1,1])
            mean_Ezt_1ztT = mean_Ezt_1ztT + Ezt_1ztT[i][t]
            mean_Ezt_1zt_1T = mean_Ezt_1zt_1T + EztztT[i][t-1]
            mean_Hvt_1 = mean_Hvt_1 + np.matmul(H,v_all[i][t-1,:].reshape([-1,1]))
            mean_Hvt_1Ezt_1T = mean_Hvt_1Ezt_1T + np.matmul(np.matmul(H,v_all[i][t-1,:].reshape([-1,1])), Ezt[i][:,t-1].reshape([1,-1]))
            
    mean_Ezt = mean_Ezt / ((T-1)*M)
    mean_Ezt_1 = mean_Ezt_1 / ((T-1)*M)
    mean_Ezt_1ztT = mean_Ezt_1ztT / ((T-1)*M)
    mean_Ezt_1zt_1T = mean_Ezt_1zt_1T / ((T-1)*M)
    mean_Hvt_1 = mean_Hvt_1 / ((T-1)*M)
    mean_Hvt_1Ezt_1T = mean_Hvt_1Ezt_1T / ((T-1)*M)
        
    tmp_1 = mean_Ezt_1ztT.T - np.matmul(mean_Ezt, mean_Ezt_1.T) - mean_Hvt_1Ezt_1T + np.matmul(mean_Hvt_1, mean_Ezt_1.T)
    tmp_2 = mean_Ezt_1zt_1T - np.matmul(mean_Ezt_1, mean_Ezt_1.T)
    
    A = np.matmul(tmp_1, np.linalg.pinv(tmp_2))
    
    ############################ b
    b = mean_Ezt - mean_Ezt_1 - mean_Hvt_1
    
    ############################ Q
    tmp = np.zeros([z_dim,z_dim])
    for i in range(M):
        for t in range(1,T):
            tmp = tmp + EztztT[i][t] - 2*np.matmul(Ezt_1ztT[i][t].T,A.T) \
            - 2*np.matmul(np.matmul(H,v_all[i][t-1,:].reshape([-1,1]))+b,Ezt[i][:,t].reshape([1,-1])) \
            + np.matmul(A, np.matmul(EztztT[i][t-1],A.T)) \
            + 2*np.matmul(np.matmul(H,v_all[i][t-1,:].reshape([-1,1]))+b,np.matmul(Ezt[i][:,t-1].reshape([1,-1]),A.T)) \
            + np.matmul(np.matmul(H,v_all[i][t-1,:].reshape([-1,1]))+b, (np.matmul(H,v_all[i][t-1,:].reshape([-1,1]))+b).T)
    Q = tmp / ((T-1)*M)
    
    ############################ C
    mean_wt = np.zeros([w_dim,1])
    mean_wtEztT = np.zeros([w_dim, z_dim])
    mean_Ezt = np.zeros([z_dim, 1])
    for i in range(M):
        for t in range(T):
            mean_wt = mean_wt + w_all[i][t,:].reshape([-1,1])
            mean_wtEztT = mean_wtEztT + np.matmul(w_all[i][t,:].reshape([-1,1]),Ezt[i][:,t].reshape([1,-1]))
            mean_Ezt = mean_Ezt + Ezt[i][:,t].reshape([-1,1])
    mean_wt = mean_wt / (T*M)
    mean_wtEztT = mean_wtEztT / (T*M)
    mean_Ezt = mean_Ezt / (T*M)
    
    tmp_1 = mean_wtEztT - np.matmul(mean_wt, mean_Ezt.T)
    tmp_2 = np.linalg.pinv(np.matmul(mean_Ezt, mean_Ezt.T))
    C = np.matmul(tmp_1, tmp_2.T)
    ############################ d
    
    d = mean_wt - np.matmul(C,mean_Ezt)
    
    ############################ R
    mean_wt_dwt_dT = np.zeros([w_dim, w_dim])
    mean_EztEztT = np.zeros([z_dim,z_dim])
    mean_wt_dEztT = np.zeros([w_dim,z_dim])
    for i in range(M):
        for t in range(T):
            mean_wt_dwt_dT = mean_wt_dwt_dT + np.matmul(w_all[i][t,:].reshape([-1,1]) - d, (w_all[i][t,:].reshape([-1,1]) - d).T)
            mean_EztEztT = mean_EztEztT + EztztT[i][t]
            mean_wt_dEztT = mean_wt_dEztT + np.matmul(w_all[i][t,:].reshape([-1,1]) - d,Ezt[i][:,t].reshape([1,-1]))
    mean_wt_dwt_dT = mean_wt_dwt_dT / (T*M)
    mean_EztEztT = mean_EztEztT / (T*M)
    mean_wt_dEztT = mean_wt_dEztT / (T*M)
    
    R = mean_wt_dwt_dT + np.matmul(C, np.matmul(mean_EztEztT.T,C.T)) \
        - 2*np.matmul(mean_wt_dEztT,C.T)
    return [A,b,H,C,d,Q,R,mu_0,Sig_0]



def EM_step(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0):
    [Ezt, EztztT, Ezt_1ztT] = expectation(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0)
    [A,b,H,C,d,Q,R,mu_0,Sig_0] = maximization(Ezt, EztztT, Ezt_1ztT, w_all, v_all)
    return [A,b,H,C,d,Q,R,mu_0,Sig_0]

T = 5
M = 1
w_all = [None]*M
v_all = [None]*M
for i in range(M):
    for t in range(T):
        w_all[i] = np.random.uniform(-1,1,3*T).reshape([T,3])
        v_all[i] = np.zeros([T,2])
    
w_dim, z_dim, v_dim = 3, 2, 2
mu_0, Sig_0 = np.zeros([z_dim,1]), np.eye(z_dim)
A,b,H,Q = np.eye(z_dim), np.zeros([z_dim,1]), np.ones([z_dim, v_dim])/v_dim, np.eye(z_dim)
C,d,R = np.ones([w_dim, z_dim])/z_dim, np.zeros([w_dim,1]), np.eye(w_dim)


kf = KalmanFilter(initial_state_mean = mu_0.reshape([-1]),
                  initial_state_covariance = Sig_0,
                  transition_matrices = A,
                  transition_offsets = b.reshape([-1]),
                  transition_covariance = Q,
                  observation_matrices = C,
                  observation_offsets = d.reshape([-1]),
                  observation_covariance = R)
kf = kf.em(w_all[0],n_iter=1,em_vars=['initial_state_mean', 'initial_state_covariance',\
           'transition_matrices', 'transition_offsets', 'transition_covariance', \
           'observation_matrices', 'observation_offsets', 'observation_covariance'])

for i in range(1):
    [A,b,H,C,d,Q,R,mu_0,Sig_0] = \
    EM_step(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0)
    
    
'''
print(mu_0.reshape([-1]))
print(kf.initial_state_mean)
print()

print(Sig_0)
print(kf.initial_state_covariance)
print()

print(A)
print(kf.transition_matrices)
print()

print(b.reshape([-1]))
print(kf.transition_offsets)
print()
'''


def log_likelihood(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0):
    
    L = 0
    
    M = len(w_all)
    T = w_all[0].shape[0]
    
    Ezt = [None] * M
    EztztT = [None] * M
    Ezt_1ztT = [None] * M
    
    z_dim = mu_0.shape[0]
    for i in range(M):
       
        Ezt[i] = np.zeros([z_dim,T])
        EztztT[i] = [None] * T
        Ezt_1ztT[i] = [None] * T
            
        
        mu_t_t = [None] * T
        Sig_t_t = [None] * T
        mu_t_t_1 = [None] * T
        Sig_t_t_1 = [None] * T
        
        for t in range(T):
            if t > 0:
                mu_t_t_1[t] = np.matmul(A,mu_t_t[t-1]) + np.matmul(H,v_all[i][t-1,:].reshape([-1,1])) + b
                Sig_t_t_1[t] = np.matmul(A, np.matmul(Sig_t_t[t-1],A.T)) + Q
            else:
                mu_t_t_1[t] = mu_0
                Sig_t_t_1[t] = Sig_0
            K = np.matmul(\
                          np.matmul(Sig_t_t_1[t], C.T) , \
                          np.linalg.pinv(\
                                        np.matmul(\
                                                  C,
                                                  np.matmul(Sig_t_t_1[t], C.T)\
                                                 )+R\
                                       )\
                         )
            mu_t_t[t] = mu_t_t_1[t] + np.matmul(K,w_all[i][t,:].reshape([-1,1]) - np.matmul(C,mu_t_t_1[t]) - d)
            Sig_t_t[t] = np.matmul(np.eye(K.shape[0]) - np.matmul(K,C), Sig_t_t_1[t])
            mu_xt_x1tot_1 = np.matmul(C,mu_t_t_1[t]) + d
            Sig_xt_x1tot_1 = np.matmul(np.matmul(C,Sig_t_t_1[t]),C.T) + R
            
            L = L - 0.5*np.log(2*np.pi*np.linalg.det(Sig_xt_x1tot_1)) -\
                0.5 * np.matmul(np.matmul(\
                                          w_all[i][t,:].reshape([1,-1]) - mu_xt_x1tot_1.T,\
                                          np.linalg.inv(Sig_xt_x1tot_1)\
                                         ),\
                                w_all[i][t,:].reshape([-1,1]) - mu_xt_x1tot_1)
            #L = L + multivariate_normal.logpdf(w_all[i][t,:],mean=mu_xt_x1tot_1.reshape([-1]),cov=Sig_xt_x1tot_1)
        
    return L



L1 = log_likelihood(w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0)
L2 = log_likelihood(w_all,\
                    kf.transition_matrices, kf.transition_offsets.reshape([-1,1]), H,v_all,\
                    kf.observation_matrices, kf.observation_offsets.reshape([-1,1]),\
                    kf.transition_covariance, kf.observation_covariance,\
                    kf.initial_state_mean.reshape([-1,1]), kf.initial_state_covariance)

print(L1)
print(L2)
        