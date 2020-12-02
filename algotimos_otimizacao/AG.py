# coding: utf-8
import math
import numpy as np
from algotimos_otimizacao.selectionOperator import Selection
from algotimos_otimizacao.funcao_AG import *
import time
import pandas as pd

# IMPLEMENTAÇÃO DE ALGORITMO GENETICO BINARIO SIMPLES
# SOLUÇÃO s de dimensão n+1, onde a primeira posição da solução representa o FO(s)
# POPULAÇÃO é uma matriz com k soluções
# Metodo de seleção é implementado em selectionOperator.py, metodo dos kMax e de torneio

class AG:
    def __init__(self, fo_func = func_obj, selection_method = Selection.Select_Tournament):
        self.best_fo = 0.0
        self.fo = fo_func
        self.best_s = []
        self.selection = selection_method

    # mutação simples, soma valor aleatorioa de -1 à 1 em uma posição aleatoria da solução
    def SimpleMutation(self, s):
        value = np.random.randint(len(s)-1)
        s[value+1] += np.random.uniform(-1,1)
        return s


    def ContinuosMutation(self, s, theta = 0.5):
        value = np.random.randint(len(s)-1)
        s[value+1] = np.random.uniform(s[value+1]-theta, s[value+1]+theta)
        return s

    def ContinuosCrossOverBLX(self, dad1, dad2, alpha):
        dif = dad1 - dad2
        son1 = np.zeros(len(dif))
        son2 = np.zeros(len(dif))

        for i in range(len(dif)):
            vmin = min([dad1[i], dad2[i]])
            vmax = max([dad1[i], dad2[i]])
            u1 = np.random.uniform(vmin - alpha*dif[i], vmax + alpha*dif[i])
            son2[i] = u1
            u2 = np.random.uniform(vmin - alpha*dif[i], vmax + alpha*dif[i])
            son2[i] = u2
        return son1, son2


    # CrossOver simples
    def SimpleCrossOver(self, dad1, dad2, alpha = 0.5):
        lim1 = int(alpha*(len(dad1)-1))
        #print(':,',lim1)
        son1 = np.concatenate((dad1[:lim1], dad2[lim1:]))
        son2 = np.concatenate((dad1[lim1:], dad2[:lim1]))
        return son1, son2

    def solve_old(self, fo = func_obj, dim_pop = 100, dim_s = 6, tx_cross = 1.0, tx_mutation = 0.2, it = 50, cross_over = SimpleCrossOver, mutation = SimpleMutation):
        seletor = Selection(fo)
        pop = seletor.RandomPopulation(dim_pop = dim_pop, dim_s = dim_s)        
        mean_pop = []
        # Loop de iteações
        for k in range(it):
            print('it:', k)
            # calcula fo pra todas soluções
            #print("Caalc FO:")
            for s in pop:
                s[0] = self.fo(s[1:])
            mean_pop.append(sum(pop[:,0])/len(pop[:,0]))
            select_pop = seletor.Select_Tournament(pop) # populção selecionada

            # lista utilizada para controlar os sorteios de forma mais eficiente
            controll_list = list(range(select_pop.shape[0])) # lista vazia do tamanho da população selecionada
            children = []

            # CROSS OVER
            for i in range(int(len(select_pop)/2)):
                if np.random.random() < tx_cross:
                    #print('Aplicando mutação:', i)
                    # sorteia pai 1
                    value1 = np.random.randint(0,len(controll_list))
                    dad1 = select_pop[controll_list[value1]]
                    controll_list.remove(controll_list[value1])

                    # sorteia pai 2
                    value1 = np.random.randint(len(controll_list))
                    dad2 = select_pop[controll_list[value1]]
                    controll_list.remove(controll_list[value1])

                    # aplica operação de crossover
                    son1, son2 = cross_over(dad1, dad2, 0.5)
                    #son1, son2 = self.SimpleCrossOver(dad1, dad2, 0.5)
                    children.append(son1)
                    children.append(son2)
                    #print('Mutação concluida')
                else:
                    children.append(select_pop[np.random.randint(len(select_pop))])
                    children.append(select_pop[np.random.randint(len(select_pop))])
                    

            # junta os filhos gerados com a populção selecionada
            pop = np.vstack((select_pop, np.array(children)))

            # mutação   
            for i in range(len(pop)):
                if np.random.random() < tx_mutation:
                    mutation(pop[i])

            # calcula os FO(s), salva melhor solução e melhor fo     
            for s in pop:
                s[0] = self.fo(s[1:])
                if s[0] < self.best_fo:
                    self.best_fo = s[0]
                    self.best_s = s[1:]
            #if k % 50 == 0:
                #print("**Report**: \nPopulation Number: ", len(pop), "\nIteration number: ", k, "\nBest fo: ", self.best_fo)
                #print("\n-------------------------------------------------------------\n")

        for s in pop:
            s[0] = self.fo(s[1:])
            if s[0] < self.best_fo:
                self.best_fo = s[0]
                self.best_s = s[1:]
        self.min_fo = min(pop[:,0])

        self.mean_fo = sum(pop[:,0])/len(pop[:,0])

        return self.best_fo, self.min_fo, self.mean_fo, mean_pop

    # otimização do algoritmo genetico
    def solve(self, fo = func_obj, dim_pop = 100, dim_s = 6, tx_cross = 0.6, tx_mutation = 0.4, it = 100, neuralnet = False, env = False, params_nn = False,  onehot_encode = False):
        cross_over = self.SimpleCrossOver
        mutation = self.ContinuosMutation
        seletor = Selection(fo)
        pop = seletor.RandomPopulation(dim_pop = dim_pop, dim_s = dim_s, neuralnet = neuralnet, env = env, params_nn = params_nn,  onehot_encode = onehot_encode)        
        mean_pop = []
        # Loop de iteações
        for k in range(it):
            print('it: ', k)
            print('***************bestfo:', self.best_fo)
            # calcula fo pra todas soluções
            #print("Caalc FO:")
            for s in pop:
                s[0] = self.fo(s[1:], neuralnet, env, params_nn, onehot_encode)
            mean_pop.append(sum(pop[:,0])/len(pop[:,0]))
            select_pop = seletor.Select_Roulette(pop) # populção selecionada

            # lista utilizada para controlar os sorteios de forma mais eficiente
            controll_list = list(range(select_pop.shape[0])) # lista vazia do tamanho da população selecionada
            children = []

            # CROSS OVER
            for i in range(int(len(select_pop)/2)):
                if np.random.random() < tx_cross:
                    #print('Aplicando mutação:', i)
                    # sorteia pai 1
                    value1 = np.random.randint(0,len(controll_list))
                    dad1 = select_pop[controll_list[value1]]
                    controll_list.remove(controll_list[value1])

                    # sorteia pai 2
                    value1 = np.random.randint(len(controll_list))
                    dad2 = select_pop[controll_list[value1]]
                    controll_list.remove(controll_list[value1])

                    # aplica operação de crossover
                    son1, son2 = cross_over(dad1, dad2, 0.5)
                    #son1, son2 = self.SimpleCrossOver(dad1, dad2, 0.5)
                    children.append(son1)
                    children.append(son2)
                    #print('Mutação concluida')
                else:
                    children.append(select_pop[np.random.randint(len(select_pop))])
                    children.append(select_pop[np.random.randint(len(select_pop))])
                    

            # junta os filhos gerados com a populção selecionada
            pop = np.vstack((select_pop, np.array(children)))

            # mutação   
            for i in range(len(pop)):
                if np.random.random() < tx_mutation:
                    mutation(pop[i])

            # calcula os FO(s), salva melhor solução e melhor fo     
            for s in pop:
                s[0] = self.fo(s[1:], neuralnet, env, params_nn, onehot_encode)
                if s[0] > self.best_fo:
                    self.best_fo = s[0]
                    self.best_s = s[1:]
                    print("Atualizou best FO")
                    w = vetor_to_weigths(s[1:], params_nn)
                    neuralnet.set_weights(w)
                    neuralnet.save('models/keras_nn_' + str(k) + '.h5')
            #if k % 50 == 0:
                #print("**Report**: \nPopulation Number: ", len(pop), "\nIteration number: ", k, "\nBest fo: ", self.best_fo)
                #print("\n-------------------------------------------------------------\n")
            
        for s in pop:
            s[0] = self.fo(s[1:], neuralnet, env, params_nn, onehot_encode)
            if s[0] < self.best_fo:
                self.best_fo = s[0]
                self.best_s = s[1:]
        self.min_fo = min(pop[:,0])

        self.mean_fo = sum(pop[:,0])/len(pop[:,0])
        return self.best_fo, self.min_fo, self.mean_fo, mean_pop



    def AutoAvaliate(self, output = 'Results.csv'): 
        tx_mutation = [0.01, 0.10, 0.20]
        tx_cross = [0.6, 0.8, 1.0]
        pop_dim = [25, 50, 100]
        it = [50, 100, 150]
        cross = [self.ContinuosCrossOverBLX, self.SimpleCrossOver]
        mutation = [self.SimpleMutation, self.ContinuosMutation]

        dic_results = {'tx_mutation':[], 'tx_cross':[], 'size':[], 'cross':[], 'mutation':[], 
                    'teste_id':[], 'best_fo':[], 'min_fo':[], 'mean_fo':[], 'time':[], 'pop':[], 'control' :[]}
        dic_history1 = {}
        dic_history2 = {}
        dic_history3 = {}
        #str(arroba)[9:-18]
        control = 0
        for txm in tx_mutation:
            for txc in tx_cross:
                for size_pop in pop_dim:
                    for it_number in it:
                        for cr in cross:
                            for  mt in mutation:
                                calc_mean_pop = np.zeros(it_number)
                                for i in range(1, 11):
                                    if(control % 100 == 0):
                                        print(control)
                                    now = time.time()
                                    best_fo, min_fo, mean_fo, mean_pop = self.solve(dim_pop = size_pop, tx_cross = txc, tx_mutation = txm, it = it_number, cross_over = cr, mutation = mt)
                                    end = time.time()
                                    calc_mean_pop = calc_mean_pop + np.array(mean_pop) 
                                    dic_results['tx_mutation'].append(txm)
                                    dic_results['tx_cross'].append(txc)
                                    dic_results['size'].append(it_number)
                                    dic_results['cross'].append(str(cr))
                                    dic_results['mutation'].append(str(mt))
                                    dic_results['teste_id'].append(i)
                                    dic_results['best_fo'].append(best_fo)
                                    dic_results['min_fo'].append(min_fo)
                                    dic_results['mean_fo'].append(mean_fo)
                                    dic_results['time'].append(end-now)
                                    dic_results['pop'].append(size_pop)
                                    dic_results['control'].append(control)
                                if it_number == 50:
                                    dic_history1[control] = calc_mean_pop/10
                                if it_number == 100:
                                    dic_history2[control] = calc_mean_pop/10
                                if it_number == 150:
                                    dic_history3[control] = calc_mean_pop/10
                                control += 1
                 
        pd.DataFrame.from_dict(dic_history1).to_csv('50.csv', index = False)
        pd.DataFrame.from_dict(dic_history2).to_csv('100.csv', index = False)
        pd.DataFrame.from_dict(dic_history3).to_csv('150.csv', index = False)
        pd.DataFrame.from_dict(dic_results).to_csv(output, index = False)
        
        return dic_results        
    


if __name__ == "__main__":
    ag = AG()
    #solution = ag.solve(cross_over = ag.ContinuosCrossOverBLX, mutation = ag.ContinuosMutation)
    #ag.AvOld()
#(5.667234501816054e-05, array([ 6.44719880e-06,  1.35690856e-05, -4.45461015e-06, -2.70691220e-05, 6.44719880e-06,  1.35690856e-05]))