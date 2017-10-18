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

x = Input(shape=(x_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
w = Dense(w_dim, activation=None)(h)

enc = Model([x],[w])

decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(x_dim, activation='sigmoid')
h_decoded = decoder_h(w)
x_bar = decoder_mean(h_decoded)

decoder_input = Input(shape=(w_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
dec = Model(decoder_input, _x_decoded_mean)

AE = Model(x,[x_bar,w,w])
#AE = Model(x,x_bar)

act_map = Sequential()
act_map.add(Dense(4, input_shape=(u_dim,), activation='relu'))
act_map.add(Dense(v_dim, activation=None))
