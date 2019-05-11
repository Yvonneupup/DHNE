# -*- coding:utf-8 -*-
#####################
# DHNE
# Network Representation Learning Method for Dynamic Heterogeneous Network
#####################

import random
from gensim.models import Word2Vec
import networkx as nx
import sys
from multiprocessing import cpu_count
import argparse
import os
import math
import time


def createHistoricalcurrentGraph(G_list, time_window, start_node, time_step):
    """
     time step is necessary because we want representation only for last time step and
     we will create the space-time graph for [time_step, time_step-1,time_step-2,...,time_step-time_window]
    """
    G_time = nx.Graph()

    for node in list(G_list[-1].nodes):

        G_neighbors = list(G_list[-1].neighbors(node))
        for i in range(len(G_neighbors)):
                G_time.add_edge(node, G_neighbors[i], weight=1.0)

    for time1 in range(1, time_window + 1):
        if start_node not in list(G_list[- time1 - 1].nodes):
            continue
        else:
            past_node=start_node
            x = -time1
            G_past = G_list[-time1 - 1]
            past_neighbors = list(G_past.neighbors(past_node))
            for i in range(len(past_neighbors)):
                G_time.add_edge(start_node, past_neighbors[i], weight=math.exp(x))
            # considering high level neighbors
            for elt in past_neighbors:
                second_order = list(G_past.neighbors(elt))
                for i in range(len(second_order)):
                    G_time.add_edge(elt, second_order[i], weight=math.exp(x))
            if past_neighbors:
                for elt in second_order:
                    third_order = list(G_past.neighbors(elt))
                    for i in range(len(third_order)):
                        G_time.add_edge(elt, third_order[i], weight=math.exp(x))

                    for elt in third_order:
                        fourth_order = list(G_past.neighbors(elt))
                        for i in range(len(fourth_order)):
                            G_time.add_edge(elt, fourth_order[i], weight=math.exp(x))
    return G_time

def random_weight(weight_data):
   
    total = sum(weight_data.values())    

    ra = random.uniform(0, total)  
    curr_sum = 0
    next_node = None
    #keys = weight_data.keys()    
    keys = weight_data.keys()        
    for k in keys:
        curr_sum += weight_data[k]             
        if ra <= curr_sum:          
            next_node = k
            break
    return next_node

def random_walk(Historicalcurrentgraph, path_length, rand=random.Random(0), start=None):
    """ Returns a truncated random walk.
        path_length: Length of the random walk.
        start: the start node of the random walk.
    """
    weight_data = {}
    G = Historicalcurrentgraph
    path = [start]
    startime = time.time()
    while len(path) < path_length:


        if (len(path)==1):
            cur = path[-1]
            neighbors = []
            if (cur[0] == 'a'):

                Allneighbors = list(G.neighbors(cur))
                for i in range(len(Allneighbors)):
                    if (Allneighbors[i][0]=='p'):
                        neighbors.append(Allneighbors[i])

            if (cur[0] == 'p'):

                Allneighbors = list(G.neighbors(cur))
                for i in range(len(Allneighbors)):

                    if (Allneighbors[i][0]=='c'):
                        neighbors.append(Allneighbors[i])
                    if (Allneighbors[i][0]=='a'):
                        neighbors.append(Allneighbors[i])

            if (cur[0] == 'c'):

                Allneighbors = list(G.neighbors(cur))
                for i in range(len(Allneighbors)):
                    if (Allneighbors[i][0]=='p'):
                        neighbors.append(Allneighbors[i])
        else:
            neighbors=[]
            pre_cur=path[-2]
            cur=path[-1]
            if (cur[0] == 'a'):

                Allneighbors = list(G.neighbors(cur))
                for i in range(len(Allneighbors)):
                    if (Allneighbors[i][0]=='p'):
                        neighbors.append(Allneighbors[i])
            if (cur[0] == 'p'):

                Allneighbors = list(G.neighbors(cur))
                for i in range(len(Allneighbors)):
                    if (pre_cur[0]=='a'):
                        if (Allneighbors[i][0]=='c'):
                            neighbors.append(Allneighbors[i])

                    if (pre_cur[0] == 'c'):
                        if (Allneighbors[i][0]=='a'):
                            neighbors.append(Allneighbors[i])

            if (cur[0] == 'c'):

                Allneighbors = list(G.neighbors(cur))
                for i in range(len(Allneighbors)):
                    if (Allneighbors[i][0]=='p'):
                        neighbors.append(Allneighbors[i])
        if len(neighbors) > 0:
            weight_data=dict()
            for i in range(len(neighbors)):
                W=G.get_edge_data(cur, neighbors[i])
                if W:
                    weights=W.values()
                    weight=list(weights)
                    weight_data[neighbors[i]] = weight[0]

            path.append(random_weight(weight_data))
        else: break

    return path

def create_vocab(G_list, num_restart, path_length, nodes, time_step, rand=random.Random(0), time_window=1):
    walks = []

    nodes = list(nodes)
    #print(len(list(nodes)))
    # number of path is equal to number of restarts per node
    rand.shuffle(nodes)
    for node in list(nodes):
        #print(node)
        G = createHistoricalcurrentGraph(G_list, time_window, node, time_step)
        for cnt in range(num_restart):#walk num
            if node not in G_list[-1]:continue
            start=time.time()
            walks.append(random_walk(Historicalcurrentgraph=G, path_length=path_length, rand=rand, start=node))
    print("Vocabulary created")
    return walks


def DHNE(input_direc, output_file, number_restart, walk_length, representation_size, time_step,
            time_window_size, workers, vocab_window_size):
    """
    This function generates representation for all nodes in space-time-graph of all nodes of graph at t=time_step
    however we will consider only representations of nodes present in graph at t = time_step
    """
    if time_window_size > time_step:
        sys.exit("ERROR: time_window_size(=" + str(time_window_size) + ") cannot be more than time_step(=" + str(
            time_step) + "):")
    G_list = [nx.read_graphml(input_direc + "/graph_" + str(i) + ".graphml") for i in
              range(time_step - time_window_size, time_step + 1)]
    # get list of nodes
    G_time=nx.Graph()
    for node in list(G_list[-1].nodes):
        G_neighbors = list(G_list[-1].neighbors(node))
        for i in range(len(G_neighbors)):
                G_time.add_edge(node, G_neighbors[i], weight=1.0)
    nodes=list(G_time.nodes)
    print("Creating vocabulary...")
    walks = create_vocab(G_list, num_restart=number_restart, path_length=walk_length, nodes=nodes,
                         rand=random.Random(0), time_step=time_step, time_window=time_window_size)
    # time-step is decremented by 1, because, time steps are from 0 to time_step-1=total time_step length

    print("Generating representation...")
    model = Word2Vec(walks, size=representation_size, window=vocab_window_size, min_count=0, workers=workers)

    model.wv.save_word2vec_format(output_file)

    print("Representation File saved: " + output_file)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', action='store', dest='dataset', help='Dataset')
    arg = parser.parse_args()
    print('dataset =', arg.dataset)

    if arg.dataset not in ["Aminer","Dblp"]:
        print("Invalid dataset.\nAllowed datsets are Aminer, Dblp")
        sys.exit(0)

    if arg.dataset == "Aminer":
        direc = "../"+arg.dataset
        max_timestep = 15
    elif arg.dataset == "Dblp":
        direc = "../"+arg.dataset
        max_timestep =18

    number_restart = 10
    walk_length = 100
    representation_size = 128
    vocab_window_size = 5
    time_window_size = 3  

    if not os.path.exists(direc+"/aminer_result"):
        os.makedirs(direc+"/aminer_result")

    workers = cpu_count()
    
    for t in range(time_window_size,max_timestep):
        #print(t)
        print("\nGenerating " + str(representation_size) + " dimension embeddings for nodes")
        time_step = t
        start = time.time()
        DHNE(input_direc=direc + "/aminer_dataset",
                output_file=direc + "/aminer_result/vector" + str(time_step) + ".txt",
                number_restart=number_restart,
                walk_length=walk_length, vocab_window_size=vocab_window_size,
                representation_size=representation_size, time_step=time_step, time_window_size=time_window_size - 1,
                workers=workers) 
