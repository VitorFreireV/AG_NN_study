from algotimos_otimizacao.AG import AG
from algotimos_otimizacao.funcao_AG import *
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
import gym
import pandas as pd
import time
  
params_nn = [16, 8, 8, 4]
neuralnet = keras.models.load_model('models/keras_nn_15.h5')


dim_pop = calc_dim(params_nn)

onehot_encode = OneHotEncoder()
onehot_encode = onehot_encode.fit(np.array(list(range(16))).reshape(-1,1))
env = gym.make("FrozenLake-v0")

new_state = onehot_encode.transform([[0]])
scores_all = 0.0
# avaliando rede neural no ambiente
env.reset()
for i in range(30):
    y = neuralnet.predict([new_state])
    action = np.argmax(y)
    env.render()
    #print("actions: ", action)
    new_state2, reward, done, info = env.step(action)
    new_state = onehot_encode.transform([[new_state2]])
    scores_all += reward
    time.sleep(1)
    
    if done:
        if scores_all == 1:
            print('ganhou: score: ', scores_all)
        #return scores_all
print("score:", scores_all)
