import sys, json, glob, re, os
import numpy as np
import pylab as plt
import networkx as nx
from networkx.readwrite import json_graph
import igraph
from sklearn import svm, metrics, preprocessing
from sklearn import cross_validation
import sys, json, glob, re, pickle, random
from scipy.stats.stats import pearsonr


names = {"BE" : "Belgium",
 "FR" : "France",
 "BA" : "Bosnia",
 "HR" : "Croatia",
 "DE" : "Germany",
 "HU" : "Hungary",
 "BU" : "Bulgaria",
 "FI" : "Finland",
 "NL" : "Netherlands",
 "PT" : "Portugal",
 "DK" : "Denmark",
 "LV" : "Latvia",
 "LT" : "Lithuania",
 "RO" : "Romania",
 "PL" : "Poland",
 "CH" : "Swiss",
 "GR" : "Greece",
 "IR" : "Ireland",
 "AL" : "Albania",
 "IT" : "Italia",
 "CZ" : "Czech-Republic",
 "AT" : "Austria",
 "CS" : "Serbia",
 "ES" : "Spain",
 "NO" : "Norway",
 "MK" : "Macedonia",
 "SK" : "Slovakia",
 "EE" : "Estonia",
 "SI" : "Slovenia",
 "UA" : "Ukraine",
 "SE" : "Sweden",
 "GB" : "United-Kingdom"}

class ExchangeGraph(object):
    def __init__(self, name):
        self.name = name
        self.digraph = None
        self.meteo = None
        self.quotes = None

    def load(self, filename):
        G = igraph.Graph(directed=True)
        self.digraph = G.Read_Pickle(filename)
        self.meteo = pickle.load(open("/Users/vallette/projects/DATA/epml/outputs/meteo.p", "rb" ))
        self.quotes = pickle.load(open("/Users/vallette/projects/DATA/epml/outputs/oil.p", "rb" ))

    def mk_mean_digraph(self, output):
        Hnx = nx.DiGraph()
        for node in self.digraph.vs:
            Hnx.add_node(node["name"])
        for node in self.digraph.vs:
            for innode in node.predecessors():
                idEdge = self.digraph.get_eid(innode.index, node.index)
                edge = self.digraph.es[idEdge]
                mean = np.mean(sum(edge.attributes().values(), []))
                Hnx.add_edge(innode["name"], node["name"], value = mean, type = "arrow_in")
                

            for outnode in node.successors():
                idEdge = self.digraph.get_eid(node.index, outnode.index)
                edge = self.digraph.es[idEdge]
                mean = np.mean(sum(edge.attributes().values(), []))
                Hnx.add_edge(node["name"], outnode["name"], value = mean, type = "arrow_out")

        #check for self loops and null
        for edge in Hnx.edges():
            if Hnx.edge[edge[0]][edge[1]]["value"] < 1.:
                print edge
                #del Hnx.edge[edge[0]][edge[1]]
            elif edge[0] == edge[1]:
                del Hnx.edge[edge[0]][edge[1]]

        self.Hnx = nx.relabel_nodes(Hnx, names)
        self.data = json_graph.node_link_data(self.Hnx)
        formated = json.dumps(self.data).replace("id", "name").replace("},", "},\n")

        out = open(output, "w")
        out.write(formated)
        out.close()


    def sankey(self, output):
        # data2 = self.data.copy()
        # data2["links"] = []

        # links = self.data["links"]
        # for link in links:
        #   source = link["source"]
        #   target = link["target"]
        #   value = link["value"]
        #   combined = False
        #   for link2 in links:
        #       if link2 != link:
        #           if source == link2["target"] and target == link2["source"]:
        #               if value - link2["value"] > 0.:
        #                   data2["links"] += [{"source" : source, "target" : target, "value" : value - link2["value"]}]
        #               else:
        #                   data2["links"] += [{"source" : target, "target" : source, "value" : link2["value"] - value}]
        #               links.remove(link2)
        #               combined = True
            # if not combined:
            #   data2["links"] += [{"source" : source, "target" : target, "value" : value}]

        G = nx.Graph(self.Hnx)
        data = json_graph.node_link_data(G)

        #check for self loops and null
        for link in data["links"]:
            if link["value"] == 0.0:
                data["links"].remove(link)
                print "removed because 0", link
            if link["target"] == link["source"]:
                data["links"].remove(link)
                print "removed because selflinked", link

        formated = json.dumps(data).replace("id", "name").replace("},", "},\n")

        out = open(output, "w")
        out.write(formated)
        out.close()


    def learn(self):

        data = np.array([], dtype="float64", order="C")
        result = np.array([], dtype="float64", order="C")

        for edge in self.digraph.es[12:16]:
            for day in edge.attribute_names():
                mean = np.mean(edge[day])
                source = self.digraph.vs[edge.source]["name"]
                target = self.digraph.vs[edge.target]["name"]
                temp_source = np.mean(self.meteo[source][day])
                temp_target = np.mean(self.meteo[target][day])
                result = np.append(result, mean)
                list = [int(day), edge.source, edge.target, temp_source, temp_target, self.quotes["OIL"][int(day)], self.quotes["KOL"][int(day)]]
                if len(data) > 0:
                    data = np.vstack((data, list))
                else:
                    data = np.array(list)


        result_n = preprocessing.scale(result)
        data_n = preprocessing.scale(data)

        n_samples = len(result)
        indexer = np.arange(n_samples)
        random.shuffle(indexer)
        result_s = result_n[indexer]
        data_s = data_n[indexer]

        size = n_samples
        classifier = svm.NuSVR(C=100)
        classifier.fit(data_s[:size], result_s[:size])

        if False:
            yp,yr = [],[]
            for i in range(0,n_samples):
                yp += [classifier.predict(data_n[i])]
                yr += [result_n[i]]
            plt.plot(range(len(yp)), yp, 'r')
            plt.plot(range(len(yr)), yr, 'b')
            plt.show()

        scores = cross_validation.cross_val_score(classifier, data_s[:size], result_s[:size], cv=3)
        print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)




europe = ExchangeGraph("Europe")
europe.load("/Users/vallette/projects/DATA/epml/outputs/graph.p")
# europe.mk_mean_digraph("/Users/vallette/projects/DATA/epml/outputs/digraph.json")
# europe.sankey("/Users/vallette/projects/DATA/epml/outputs/graph.json")

europe.learn()









def smooth(x,window_len=11,window='hanning'):
    x = np.array(x)
    s = np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': 
        w = ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]



