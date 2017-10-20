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
from Kalman_tools import expectation, maximization, EM_step, E_log_P_x_and_z, KF_predict

#np.random.seed(423)

w_dim, z_dim, v_dim, x_dim, u_dim = 4, 4, 4, 2, 2
intermediate_dim = 4
    
mu_0, Sig_0 = np.zeros([z_dim,1]), np.eye(z_dim)
A,b,H,Q = np.eye(z_dim) + np.random.uniform(-0.1,0.1,z_dim*z_dim).reshape([z_dim,z_dim]), np.zeros([z_dim,1]), np.ones([z_dim, v_dim])/v_dim, np.eye(z_dim)
C,d,R = np.ones([w_dim, z_dim])/z_dim + np.random.uniform(-0.1,0.1,w_dim*z_dim).reshape([w_dim,z_dim]), np.zeros([w_dim,1]), np.eye(w_dim)

