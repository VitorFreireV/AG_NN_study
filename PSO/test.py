from pso import PSO
from funcao_AG import *
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import gym
import pandas as pd

# rede com 16 entradas, 16 neuronios, 16, 4
def build_model(params_nn):
    model = Sequential()
    model.add(Dense(params_nn[1], input_dim=params_nn[0], activation='relu'))
    model.add(Dense(params_nn[2], activation = 'relu'))
    model.add(Dense(params_nn[3], activation='softmax'))
    return model


params_nn = [16, 8, 8, 4]
neuralnet = build_model(params_nn)

dim_pop = calc_dim(params_nn)

onehot_encode = OneHotEncoder()
onehot_encode = onehot_encode.fit(np.array(list(range(16))).reshape(-1,1))

env = gym.make("FrozenLake-v0")

pso = PSO()

G, best_fo, results = pso.solve(dim = dim_pop, neuralnet = neuralnet, env = env, params_nn = params_nn, onehot_encode = onehot_encode)