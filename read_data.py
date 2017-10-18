print('data load start')
[tmp_X, tmp_U, tmp_S] = pickle.load(open('moving_particle_trajectory_train', 'rb'))
print('data load finish')

x_all = [None] * len(tmp_X)
u_all = [None] * len(tmp_U)
for i in range(len(tmp_X)):
    x_all[i] = tmp_X[i].reshape([tmp_X[i].shape[0],-1])
    x_all[i] = x_all[i] / 100
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
print('train data is ready.')

print('data load start')
[tmp_X, tmp_U, _] = pickle.load(open('moving_particle_trajectory_test', 'rb'))
print('data load finish')

x_test_all = [None] * len(tmp_X)
u_test_all = [None] * len(tmp_U)
for i in range(len(tmp_X)):
    x_test_all[i] = tmp_X[i].reshape([tmp_X[i].shape[0],-1])
    x_test_all[i] = x_test_all[i] / 100
    u_test_all[i] = tmp_U[i][:-1,:].reshape([tmp_U[i].shape[0]-1,-1])

x_test = x_test_all[0]
for i in range(1,len(x_test_all)):
    x_test = np.concatenate([x_test, x_test_all[i]])
print('test data is ready.')


EzT_CT_Rinv_plus_dT_Rinv = np.zeros([x_train.shape[0],w_dim])

u_train = u_all[0]
for i in range(1,len(u_all)):
    u_train = np.concatenate([u_train, u_all[i]])
EztT_minus_Ezt_1TAT_bT_alltimes_QinvH = np.zeros([u_train.shape[0],v_dim])

u_test = u_test_all[0]
for i in range(1,len(u_test_all)):
    u_test = np.concatenate([u_test, u_test_all[i]])

w_all = [None] * len(x_all)
v_all = [None] * len(u_all)
