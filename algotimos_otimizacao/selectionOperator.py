import numpy as np
from algotimos_otimizacao.funcao_AG import func_obj

class Selection:
    def __init__(self, fo = func_obj):
        self.fo = fo
    # dim_pop = dimensão da populção
    # sim_pop dimensão de uma unica solução
    # Retorna matriz com dim_pop x dim_s, onde cada valor da solução tem chance de ser um random entre serch_interval
    def RandomPopulation(self, dim_pop = 100, dim_s = 6, search_interval = [-1.0, 1.0], neuralnet = False, env = False, params_nn = False,  onehot_encode = False):
        pop = np.random.uniform(search_interval[0], search_interval[1], size = (dim_pop, dim_s+1))
        for i in range(pop.shape[0]):
            pop[i][0] = self.fo(pop[i][1:], neuralnet, env, params_nn, onehot_encode)            
        return pop

    # seleciona metade da população com maior fo
    def Select_kMax(self, pop):
        half_number = int(pop.shape[0]/2)
        selected_fo = sort(pop[:][0])[half_number:]

        list_selectd_s = []
        for i in range(pop.shape[0]):
           if pop[i,0] in selected_fo:
               list_selectd_s.append(pop[i])
               if(len(list_selectd_s) == half_number):
                   return np.array(list_selectd_s)

        return np.array(list_selectd_s)

    # metodo de seleção por torneio, retorna sempre a metade da população
    def Select_Tournament(self, pop):
        candidates = list(range(0, pop.shape[0]))
        half_number = int(pop.shape[0]/2)
        selected_candidates = []

        for i in range(half_number):

            value1 = np.random.randint(0, len(candidates))
            candidate1 = candidates[value1]
            candidates.remove(candidate1)

            value2 = np.random.randint(0, len(candidates))
            candidate2 = candidates[value2]
            candidates.remove(candidate2)

            if(pop[candidate1][0] < pop[candidate2][0]):
                selected_candidates.append(candidate1)
            else:
                selected_candidates.append(candidate2)

        new_pop = []
        for s in selected_candidates:
            new_pop.append(pop[s])
        #print('Fim seleção')
        return np.array(new_pop)


    def Select_Roulette(self, pop):
        candidates = list(range(0, pop.shape[0]))
        pop_size = int(pop.shape[0]/2)
        selected_candidates = []
        #print('Selecao: ')

        sum_fitnes = np.sum(pop[:,1])
        if (sum_fitnes == 0.0):
            #print("FITNESS 0 DE POPULAÇÃO")
            sum_fitnes = 0.000000000001
        addeds = []
        roulette = pop[:,1]/sum_fitnes
        #print(pop)
        for i in range(pop_size):
            value = np.random.random()
            counter = 0.0
            for j in range(len(roulette)):
                counter += roulette[j]
                # mais elitismo - sem retirar ja sorteado

                if counter > pop[j][0]:
                    selected_candidates.append(pop[j])
                    addeds.append(j)
                    break
        #print('Fim Selecao: ')
        return np.array(selected_candidates)                    


    def Select_Roulette2(self, pop):
        candidates = list(range(0, pop.shape[0]))
        pop_size = int(pop.shape[0]/2)
        selected_candidates = []
        #print('Selecao: ')

        sum_fitnes = np.sum(pop[:,0])
        if (sum_fitnes == 0.0):
            #print("FITNESS 0 DE POPULAÇÃO")
            sum_fitnes = 0.000000000001
        addeds = []
        roulette = pop[:,0]/sum_fitnes
        #print(pop)
        for i in range(pop_size):
            value = np.random.random()
            counter = 0.0
            for j in range(len(roulette)):
                counter += roulette[j]
                # mais elitismo - sem retirar ja sorteado
                if j not in addeds:
                    if counter > pop[j][0]:
                        selected_candidates.append(pop[j])
                        addeds.append(j)
                        break
        #print('Fim Selecao: ')
        return np.array(selected_candidates)                    
