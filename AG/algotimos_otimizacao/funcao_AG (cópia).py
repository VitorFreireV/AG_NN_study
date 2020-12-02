# coding: utf-8
import math
import numpy as np

def calc_dim(param_nn = [1,2,3,4]):
    dim = 0
    for i in range(1, len(param_nn)):
        dim += param_nn[i-1]*param_nn[i] + param_nn[i]
    return dim

def vetor_to_weigths(vetor, param_nn = [1, 2, 3, 4]):
    weigths_list = []
    pivo = 0
    for i in range(1, len(param_nn)):
        weigths = vetor[pivo:pivo+param_nn[i - 1]*param_nn[i]]
        weigths = weigths.reshape(param_nn[i-1], param_nn[i])
        weigths_list.append(weigths)
        pivo = param_nn[i - 1]*param_nn[i]
        bias = vetor[pivo:pivo+param_nn[i]]
        pivo += param_nn[i]
        weigths_list.append(bias)
    return weigths_list

def func_obj(solution, neuralnet, env, params_nn, onehot_encode):
    w = vetor_to_weigths(solution, params_nn)
    neuralnet.set_weights(w)
    new_state = onehot_encode.transform([[0]])
    scores_all = 0.0
    # avaliando rede neural no ambiente
    env.reset()
    for i in range(30):
        y = neuralnet.predict([new_state])
        action = np.argmax(y)
        #print("actions: ", action)
        new_state2, reward, done, info = env.step(action)
        new_state = onehot_encode.transform([[new_state2]])
        scores_all += reward
        
        if done:
            #print(new_state2)
            return new_state2
    if scores_all > 0 :
        #print(scores_all*new_state2)
        return scores_all*new_state2
    else:
        # calc posição para dar F0 com base na distancia
        row = int(new_state2/4)
        col = int(new_state2%4)
        result = np.sqrt((4-row)**2 + (4-col)**2)
        result = 1/result
        #print(new_state2)
        return new_state2


