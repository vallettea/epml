import time, sys, csv, random, json,sys
from collections import OrderedDict
import matplotlib
import numpy as np
import pylab as plt
from scipy.stats.stats import pearsonr
from sklearn import svm, metrics, preprocessing
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier
import pandas
from deap import algorithms, base, creator, tools


class Variable(object):
    def __init__(self, name, description, col, type):
        self.name = name
        self.description = description
        self.col = col
        self.type = type

class DataStructure(object):
    def __init__(self, filename, output_name):
        self.filename = filename
        self.variables = {}
        self.output_name = output_name
        self.output = None

    def load(self):
        with open(self.filename, "rb") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",")
            next(spamreader) # skip first line
            for row in spamreader:
                v = Variable(row[0], row[1], int(row[2]) - 1 , row[3]) # carefull: python starts at zero
                self.variables[row[0]] = v

        for v in self.variables.values():
            if v.name == self.output_name:
                self.output = v
                break

        # remove imputation flags (rubish data starting by Z)and non numeric values
        self.variables = {k:v for (k,v) in self.variables.items() if v.name[0] != "Z" and v.type == "Numeric"}
        # remove all variables from col 389 (these are the resuslts)
        self.variables = {k:v for (k,v) in self.variables.items() if (v.col < 840 or v.name==self.output_name)}



class DataSet(object):
    def __init__(self, filename, data_structure):
        self.filename = filename
        self.data_structure = data_structure
        self.data = np.array([], dtype = "float64", order = "C")
        self.result = np.array([], dtype = "float64", order = "C")
        self.feature_names = []
        self.nb_samples = 0
        self.nb_features = 0
        self.selected_features = []

    def load_data(self):
        columns = [f.col for f in self.data_structure.variables.values()]
        self.data = pandas.read_csv("data/recs2009_public_v3.csv", delimiter=",", usecols=columns)
        self.data = self.data.fillna(0) # missing values are replaced by zeros (here ther is only 4 of them)
        self.feature_names = self.data.columns
        self.data = self.data.apply(np.float64)
        self.result = self.data.pop(self.data_structure.output_name)
        self.nb_samples, self.nb_features = self.data.shape

    def normalize(self):
        self.data = pandas.DataFrame(preprocessing.scale(self.data), index = self.data.index, columns = self.data.columns)
        self.result = preprocessing.scale(self.result)

        # shuffle
        indexer = np.arange(self.nb_samples)
        random.shuffle(indexer)
        self.data = self.data.ix[indexer]
        self.result = self.result[indexer]

    def del_correlate(self):
        corr = self.data.corr()
        for name in corr.columns:
            for name2 in corr.index:
                if corr[name][name2] > 0.9:
                    try:
                        del self.data[name2]
                        self.data_structure[name].description += " SAME AS " + self.data_structure[name2].description
                    except:
                        pass
                    break
        self.feature_names = self.data.columns
        self.nb_samples, self.nb_features = self.data.shape


    def learn(self, use_feature, silent= False, sample_size=1000):
        classifier = svm.NuSVR()
        if silent: # in silent mode it is a chromosom so ints
            use_data = np.array(self.data[self.data.columns[use_feature]])
        else:
            use_data = np.array(self.data[use_feature])
        classifier.fit(use_data[:sample_size], self.result[:sample_size])
        expected = self.result[sample_size:2*sample_size]
        predicted = classifier.predict(use_data[sample_size:2*sample_size])
        if silent:
            #return pearsonr(expected,predicted)[0]
            scores = cross_validation.cross_val_score(classifier, use_data[:sample_size], self.result[:sample_size], cv=3)
            return scores.mean()
        else:
            scores = cross_validation.cross_val_score(classifier, use_data[:self.nb_samples/2], self.result[:self.nb_samples/2], cv=3)
            print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
            return scores.mean()


    def feature_importance(self, use_feature, sample_size=1000, nb_selected_features = 12, plot = False):
        forest = ExtraTreesClassifier(n_estimators=15, compute_importances=True, random_state=0, n_jobs=-1)
        use_data = np.array(self.data[use_feature])
        forest.fit(use_data[:sample_size], self.result[:sample_size])
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]


        print "Feature ranking:"
        for f in xrange(len(use_feature)):
            print "%d. feature %s : %s (%f)" % (f + 1, self.feature_names[indices[f]], self.data_structure.variables[self.feature_names[indices[f]]].description, importances[indices[f]])

        if plot:
            fig = plt.figure()
            plt.title("Feature importances")
            ax = fig.add_subplot(111)
            ax.bar(xrange(len(use_feature)), importances, color="r", yerr=std, align="center")
            plt.xticks(xrange(len(use_feature)), [self.feature_names[i] for i in range(len(indices))],rotation=45)

            plt.xlim([-1, len(use_feature)])
            fig = plt.gcf()
            fig.set_size_inches(6, 6)
            plt.savefig("figs/feature_histogram.png")
            plt.show()

        self.selected_features = [use_feature[j] for j in indices]


class Genetic_Features(object):
    def __init__(self, data_set, gene_length =10, sample_size=1000):
        self.data_set = data_set
        self.gene_length = gene_length
        self.sample_size = sample_size
        self.used_features = []
        self.current_best = None

    def evolve(self):

        def pearson_score(individual):
            return self.data_set.learn(list(individual), sample_size=self.sample_size, silent= True),

        def crossover(ind1, ind2):
            # crossover is exchange of two allele with no repetition in the chromosoms
            for i in range(self.gene_length/2):
                found = False
                ind1 = random.sample(ind1, len(ind1))
                for posx,x in enumerate(ind1):
                    if x not in ind2:
                        found = True
                        break
                if found:
                    posy = random.randint(0, len(ind2)-1)
                    ind1[posx] = ind2[posy]
                    ind2[posy] = x
            return ind1, ind2

        def mutate(individual, indpb):
            # mutation is replacing a feature by a unused one
            len(self.data_set.data.columns)
            size = len(individual)
            for i in xrange(size):
                if random.random() < indpb:
                    unused = filter(lambda x:x not in set(self.used_features), xrange(len(self.data_set.data.columns)))
                    random.shuffle(unused)
                    if len(unused) > 0:
                        individual[i] = unused[0]
                        self.used_features += [unused[0]]
            return individual,

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='i', fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # how a gene is initialized (here a sequence of random indices of the features)
        toolbox.register("indices", random.sample, xrange(len(self.data_set.data.columns)), self.gene_length)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)


        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=4)
        toolbox.register("evaluate", pearson_score)

        random.seed(169)

        pop = toolbox.population(n=100)
        self.used_features = list(set(sum([list(x) for x in pop],[])))

        CXPB, MUTPB, NGEN = 0.7, 0.7, 50
        
        # evaluate the entire population
        fitnesses = map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            print "=============>  Generation %s" % str(g)
            # select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = map(toolbox.clone, offspring)
            
            # apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # replace population by offsprings
            pop[:] = offspring
            fits = [ind.fitness.values[0] for ind in pop]
            
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            feature_names = [self.data_set.data.columns[x] for x in invalid_ind[np.argmax(fitnesses)]] 
            self.used_features = list(set(sum([list(x) for x in pop],[])))
            self.current_best = feature_names 
            # if all feature are used stop mutation 
            if len(self.used_features) == len(self.data_set.data.columns):
                MUTPB = 0.
            for name in feature_names:       
                print "# " + self.data_set.data_structure.variables[name].description
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            print("  Used features %s/%s" % (str(len(self.used_features)), str(len(self.data_set.data.columns))))

            if std < 1e-3:
                break
        return self.current_best        








data_structure = DataStructure("data/Public File v3 layout file.csv", "CUFEETNG")
data_structure.load()

data_set = DataSet("data/recs2009_public_v3.csv", data_structure)
data_set.load_data()
data_set.normalize()
data_set.del_correlate()

# select best features from all columns
data_set.feature_importance(list(data_set.data.columns), sample_size=1000, nb_selected_features = 12, plot = False)
data_set.learn(data_set.selected_features)


# #genetic version lest efficient but i keep it as an example
# evolution = Genetic_Features(data_set, gene_length=12, sample_size=1000)
# evolution.evolve()
# data_set.learn(evolution.current_best)
# data_set.feature_importance(use_feature = evolution.current_best, plot=True)




# import cProfile
# cProfile.run('evolution.evolve()', "profiling.txt")
# import pstats
# p = pstats.Stats('profiling.txt')
# print p.strip_dirs().sort_stats(-1).print_stats()


#["HDD30YR","REGIONC", "Climate_Region_Pub", "CDD65", "TYPEHUQ", "CDD30YR", "DIVISION", "HDD65", "REPORTABLE_DOMAIN", "AIA_Zone", "KOWNRENT", "NWEIGH"]








