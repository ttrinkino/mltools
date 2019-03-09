from tensorflow.contrib.rnn import LSTMCell
import tensorflow as tf
from utils import paramData, getData
from ANN_multi import ANN
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, f1_score
from operator import itemgetter
from datetime import datetime as dt
import copy


def main():
    t0 = dt.now()

    # Filter Hypers
    filename = 'D:/data_files/NN_Imbal.feather'

    # GA Hypers
    generations = 10
    population_size = 40

    ga = GA(
        generations=generations,
        population_size=population_size
    )

    best_individual, pop = ga.fit(filename=filename)

    t1 = dt.now()
    print('----')
    print(t1 - t0)


class GA:
    def __init__(self, generations=10, population_size=200, retain=0.2, random_select=0.1, mutate=0.1):
        self.generations = generations
        self.population_size = population_size
        self.retain = int(retain * population_size)
        self.random_select = int(random_select * population_size)
        self.mutate = int(mutate * population_size)
        assert (mutate < retain + random_select)

    def fit(self, filename):
        self.filename = filename
        fext = filename.split('.csv')[0].split('.feather')[0].split('/')[-1]

        # make initial population
        pop, unique = self.population(self.population_size)
        best_individual = dict()
        for i in range(self.generations):
            t0 = dt.now()
            i += 1
            pop, best_individual, unique = self.evolve(
                pop=pop, unique=unique
            )
            num = 0
            df = dict()
            for child in pop:
                base = dict()
                for group in child.keys():
                    if type(child[group]) != dict:
                        base[group] = child[group]
                    else:
                        for hyper in child[group].keys():
                            base[hyper] = child[group][hyper]
                df[num] = base
                num += 1
            pd.DataFrame(df).T.to_csv(
                './data_files/' + str(fext) + '_savedhypers' + str(i) + '.csv'
            )
            print('Generation Number', i, 'Best Score', round(best_individual['score'], 5), dt.now() - t0)

        for indv in sorted(pop, key=itemgetter('score'), reverse=True)[:10]:
            p = 'Score: ' + str(round(indv['score'], 5))
            for group in indv.keys():
                if type(indv[group]) == dict:
                    for hyper in indv[group].keys():
                        p += ', ' + str(hyper) + ': '
                        p += str(indv[group][hyper])

        print('GA Done')

        return best_individual, pop

    def population(self, num):
        pop = list()
        unique = [None]
        n = 0
        while len(pop) < num:
            n += 1
            if n > num * 100:
                break
            child = self.individual()
            if child['hypers'] not in unique:
                unique.append(child['hypers'])
                pop.append(child)

        return pop, unique

    @staticmethod
    def individual():
        param_ranges = paramData(nn_params=True)

        child = dict()
        for group in param_ranges.keys():
            child[group] = dict()
            if group == 'fixed_params':
                for hyper in param_ranges[group].keys():
                    child[group][hyper] = np.random.choice(param_ranges[group][hyper])
            elif group == 'cell_params':
                rnn_cell = np.random.choice(param_ranges[group]['rnn_cell'])
                child[group]['rnn_cell'] = rnn_cell
                if rnn_cell == LSTMCell:
                    child[group]['use_peepholes'] = np.random.choice(param_ranges[group]['use_peepholes'])
            elif group == 'cost_params':
                cost_func = np.random.choice(param_ranges[group]['cost_func'])
                child[group]['cost_func'] = cost_func
                if cost_func in [tf.train.AdamOptimizer, tf.train.RMSPropOptimizer]:
                    child[group]['beta1'] = np.random.choice(param_ranges[group]['beta1'])
                    child[group]['beta2'] = np.random.choice(param_ranges[group]['beta2'])
                    child[group]['epsilon'] = np.random.choice(param_ranges[group]['epsilon'])
            else:
                layer_hyper = [hyper for hyper in param_ranges[group].keys() if 'layers' in hyper][0]
                neuron_hyper = [hyper for hyper in param_ranges[group].keys() if 'neurons' in hyper][0]
                layers = np.random.choice(param_ranges[group][layer_hyper])
                child[group][layer_hyper] = layers
                neuron_list = list()
                for _ in range(layers):
                    if neuron_list:
                        max_val = neuron_list[-1]
                        min_val = 0.25 * max_val
                        neurons = [i for i in param_ranges[group][neuron_hyper] if i <= max_val and i >= min_val]
                    else:
                        neurons = param_ranges[group][neuron_hyper]
                    neuron = int(float(np.random.choice(neurons)))
                    neuron_list.append(neuron)
                child[group][neuron_hyper] = neuron_list

        child['score'] = -np.inf
        child['secondary_score'] = -np.inf
        child['hypers'] = ''
        for group in child.keys():
            if type(child[group]) == dict:
                for hyper in child[group].keys():
                    child['hypers'] += str(child[group][hyper])

        return child

    def evolve(self, pop, unique):
        retain = self.retain
        random_select = self.random_select
        mutate = self.mutate

        # Grade Individuals
        graded = list()
        data = list()
        for child in pop:
            if child['score'] > -np.inf:
                graded.append(child)
            else:
                data.append(child)

        data_list = list()
        for child in data:
            data_list.append(self.fitness(child=child))
            print('Child Number', len(data_list), 'done, score:', data_list[-1]['score'])
        graded.extend(data_list)

        graded = sorted(graded, key=itemgetter('score'), reverse=True)
        parents = copy.deepcopy(graded[:retain])
        best_individual = copy.deepcopy(parents[0])

        # Random Select
        if len(graded[retain:]) > random_select:
            idxs = np.random.choice(range(len(graded[retain:])), random_select, replace=False)
            for i in idxs:
                parents.append(graded[retain:][i])
        else:
            parents.extend(graded[retain:])

        # Mutate
        idxs = np.random.choice(range(len(parents)), mutate)
        for i in idxs:
            child = copy.deepcopy(parents[i])
            n = 0
            while True:
                n += 1
                if n > len(pop) * 1000:
                    break
                child = self.mutate_child(child)
                if child['hypers'] not in unique:
                    unique.append(child['hypers'])
                    parents.append(child)
                    break

        # Crossover
        n = 0
        while len(parents) < len(pop):
            n += 1
            if n > len(pop) * 1000:
                break
            male = np.random.randint(len(parents))
            female = np.random.randint(len(parents))
            if male != female:
                male = copy.deepcopy(parents[male])
                female = copy.deepcopy(parents[female])
                child = self.crossover(male, female)
                if child['hypers'] not in unique:
                    unique.append(child['hypers'])
                    parents.append(child)

        return parents, best_individual, unique

    @staticmethod
    def crossover(male, female):
        child = copy.deepcopy(male)
        for group in child.keys():
            if group == 'fixed_params':
                for hyper in child[group].keys():
                    if np.random.randn() < 0:
                        child[group][hyper] = female[group][hyper]
            elif type(group) == dict:
                if np.random.randn() < 0:
                    child[group] = female[group]

        child['score'] = -np.inf
        child['secondary_score'] = -np.inf
        child['hypers'] = ''
        for group in child.keys():
            if type(child[group]) == dict:
                for hyper in child[group].keys():
                    child['hypers'] += str(child[group][hyper])

        return child

    def mutate_child(self, child, mutate_prob=0.25):
        new_child = self.individual()
        mutate_prob *= 100

        for group in new_child.keys():
            if group == 'fixed_params':
                for hyper in new_child[group].keys():
                    if np.random.choice(100) < mutate_prob:
                        continue
                    new_child[group][hyper] = child[group][hyper]
            elif type(group) == dict:
                if np.random.choice(100) < mutate_prob:
                    continue
                new_child[group] = child[group]

        return new_child

    def fitness(self, child):

        fit_score, secondary_score = self.judge(
            child=child
        )
        child['score'] = fit_score
        child['secondary_score'] = secondary_score

        return child

    def judge(self, child):
        data = getData(
            self.filename, load_file=True
        )

        max_seconds = child['fixed_params']['max_seconds']
        epochs = child['fixed_params']['epochs']
        learning_rate = child['fixed_params']['learning_rate']
        batch_norm_decay = child['fixed_params']['batch_norm_decay']
        fc_keep_probs = child['fixed_params']['fc_keep_probs']
        rnn0_keep_prob = child['fixed_params']['rnn0_keep_prob']
        rnn1_keep_prob = child['fixed_params']['rnn1_keep_prob']
        final_keep_probs = child['fixed_params']['final_keep_probs']
        activation_func = child['fixed_params']['activation_func']
        rnn_activation_func = child['fixed_params']['rnn_activation_func']
        batch_sz = child['fixed_params']['batch_sz']
        cost_func = child['cost_params']['cost_func']
        rnn_cell = child['cell_params']['rnn_cell']
        fc_layer_sizes = child['fc_layer_params']['fc_neurons']
        rnn0_layer_sizes = child['rnn0_layer_params']['rnn0_neurons']
        rnn1_layer_sizes = child['rnn1_layer_params']['rnn1_neurons']
        final_layer_sizes = child['final_layer_params']['final_neurons']
        if 'beta1' in child['cost_params'].keys():
            beta1 = child['cost_params']['beta1']
            beta2 = child['cost_params']['beta2']
            epsilon = child['cost_params']['epsilon']
        else:
            beta1 = None
            beta2 = None
            epsilon = None
        if 'use_peepholes' in child['cell_params'].keys():
            use_peepholes = child['cell_params']['use_peepholes']
        else:
            use_peepholes = None

        try:
            ann = ANN(
                rnn0_layer_sizes=rnn0_layer_sizes, rnn1_layer_sizes=rnn1_layer_sizes,
                fc_layer_sizes=fc_layer_sizes, final_layer_sizes=final_layer_sizes,
                rnn0_keep_prob=rnn0_keep_prob, rnn1_keep_prob=rnn1_keep_prob,
                fc_keep_probs=fc_keep_probs, final_keep_probs=final_keep_probs
            )
            train_costs, test_costs, train_accuracies, accuracies, y_pred = ann.fit(
                data, learning_rate=learning_rate, cost_func=cost_func, beta1=beta1, beta2=beta2,
                epsilon=epsilon, rnn_cell_type=rnn_cell, activation=activation_func,
                rnn_activation=rnn_activation_func, batch_norm_decay=batch_norm_decay,
                use_peepholes=use_peepholes, epochs=epochs, max_seconds=max_seconds,
                batch_sz=batch_sz, show_fig=False, return_scores=True,
                return_pred=True, print_progress=False, reset_graph=True
            )

            probs = 1 / (1 + np.exp(-y_pred))
            probs = probs[:, 1] / np.sum(probs, axis=1)
            y_test = data['y_test']
            fit_score = -log_loss(y_test, probs)
            secondary_score = np.mean(accuracies[-5:]) - np.mean(train_accuracies[-5:])

        except:
            fit_score = -1000.
            secondary_score = -1000.

        return fit_score, secondary_score


if __name__ == '__main__':
    main()
