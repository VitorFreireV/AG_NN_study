import numpy as np 
from progress.bar import IncrementalBar
import pandas as pd
from funcao_AG import *
# f_29 do artigo
'''
def fo_29(x_vet):
    result = (2*(x_vet[0]**2)) - (1.05*(x_vet[0]**4)) + ((x_vet[0]**6)/6) + x_vet[0]*x_vet[1] + x_vet[1]**2
    
    return result
'''


class PSO:
    def __init__(self, fo = func_obj, pop_size = 100, dim = 2):
        self.fo = fo


    def init_pop(self, search_interval, pop_size, dim, neuralnet = False, env = False, params_nn = False, onehot_encode = False):
        #self.X = np.zeros((pop_size, dim)) # pop inicial
        self.V = np.zeros((pop_size, dim)) # Vetor de velocidade para população
        self.P = np.ones((pop_size, dim)) # Vetor de melhor posição local
        self.G = np.ones((dim)) # Vetor de melhor posição global
        self.pop_fo = np.zeros((pop_size))
        self.X = np.array(np.random.uniform(search_interval[0], search_interval[1], size = (pop_size, dim)))
        self.P = self.X.copy()
        #print("X\n", self.X)
        #print("G\n", self.G)
        self.best_fo = 0.0

        # update G 
        for i in range(self.X.shape[0]):
            if  self.pop_fo[i] > self.best_fo:
                self.G = self.X[i].copy()
                self.best_fo = self.fo(self.X[i], neuralnet, env, params_nn, onehot_encode)



    def solve(self, it_number = 10, pop_size = 100, dim = 2 , w = 0.3, c1 = 0.5, c2 = 0.5, valid_interval = [-1, 1], neuralnet = False, env = False, params_nn = False, onehot_encode = False):
        results = {'max_fo':[], 'min_fo':[], 'mean_fo':[], 'best_s_it': 0}
        self.init_pop(valid_interval, pop_size, dim, neuralnet, env, params_nn, onehot_encode)
        for j in range(it_number):
            print('itnumber: ', j)
            print("BEST FO = ", self.best_fo)
            print("Atualizou best FO")

            weigths = vetor_to_weigths(self.G, params_nn)
            neuralnet.set_weights(weigths)
            neuralnet.save('models/keras_nn_' + str(j) + '.h5')
            for i in range(self.X.shape[0]):
                # atualiza P e G
                self.pop_fo[i] = self.fo(self.X[i], neuralnet, env, params_nn, onehot_encode)
                if self.pop_fo[i] > self.fo(self.P[i], neuralnet, env, params_nn, onehot_encode):
                    self.P[i] = self.X[i].copy()
                    if self.pop_fo[i] > self.best_fo:
                        self.best_fo = self.pop_fo[i]
                        self.G = self.X[i].copy()
                        self.best_s_it = j
            
                # atualiza V
                r1 = np.random.random()
                r2 = np.random.random()
                self.V[i] = w*self.V[i] + (r1*c1)*(self.P[i] - self.X[i]) + (r2*c2)*(self.G - self.X[i])
                self.X[i] += self.V[i]
                # validando X
                if self.X[i].max() > valid_interval[1]:
                    self.X[i][ self.X[i].argmax() ] = valid_interval[1]
                if self.X[i].min() < valid_interval[0]:
                    self.X[i][ self.X[i].argmin() ] = valid_interval[0]

            print("FO global: ", self.pop_fo.mean())
            results['max_fo'].append(self.pop_fo.max())
            results['min_fo'].append(self.pop_fo.min())
            results['mean_fo'].append(self.pop_fo.mean())
            


        return self.G, self.best_fo, results

    def fat_experiment(self):
        it_list = [100, 200, 300]
        c1_list = [0.2, 0.4, 0.6]
        c2_list = [0.2, 0.4, 0.6]
        w_list = [0.3, 0.6, 0.9]
        pop_list = [100, 200, 300]
        data = {'id':[], 'it_number': [], 'size':[], 'c1':[], 'c2':[], 'w':[], 'best_fo':[], 'best_s_it':[]}
        id = 0
        
        data_100= {'id':[], 'min_fo':[], 'max_fo':[], 'mean_fo':[]}
        data_200 = {'id':[], 'min_fo':[], 'max_fo':[], 'mean_fo':[]}
        data_300 = {'id':[], 'min_fo':[], 'max_fo':[], 'mean_fo':[]}
        bar = IncrementalBar('Processing', max= (3**5)*10)
        for it in it_list:
            for c1 in c1_list:
                for c2 in c2_list:
                    for w in w_list:
                        for size in pop_list:
                            for i in range(10):
                                best_s, best_fo, results = self.solve(it_number = it, pop_size =size, c1 = c1, c2 = c2, w = w)
                                data['id'].append(id)
                                data['it_number'].append(it)
                                data['size'].append(size)
                                data['c1'].append(c1)
                                data['c2'].append(c2)
                                data['w'].append(w)
                                data['best_s_it'].append(results['best_s_it'])
                                data['best_fo'].append(best_fo)
                       

                                if it == 100:
                                    for i in range(len(results['min_fo'])):
                                        data_100['id'].append(id)
                                        data_100['min_fo'].append(results['min_fo'][i])
                                        data_100['max_fo'].append(results['max_fo'][i])
                                        data_100['mean_fo'].append(results['mean_fo'][i])
                                    #print('data',data_100)
                                        
                                if it == 200:
                                    for i in range(len(results['min_fo'])):
                                        data_200['id'].append(id)
                                        data_200['min_fo'].append(results['min_fo'][i])
                                        data_200['max_fo'].append(results['max_fo'][i])
                                        data_200['mean_fo'].append(results['mean_fo'][i])

                                if it == 300:
                                    for i in range(len(results['min_fo'])):
                                        data_300['id'].append(id)
                                        data_300['min_fo'].append(results['min_fo'][i])
                                        data_300['max_fo'].append(results['max_fo'][i])
                                        data_300['mean_fo'].append(results['mean_fo'][i])
                                bar.next()
                            id += 1

        pd.DataFrame.from_dict(data).to_csv('results/header.csv', index = False)
        pd.DataFrame.from_dict(data_100).to_csv('results/history_100.csv', index = False)
        pd.DataFrame.from_dict(data_200).to_csv('results/history_200.csv', index = False)
        pd.DataFrame.from_dict(data_300).to_csv('results/history_300.csv', index=False)
        bar.finish()

