import pickle
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import os
from IPython.display import clear_output
from functions import one_hot
from ETM_raw.scripts.data_preprocessing import *
rng = default_rng(seed = 0)

def get_scenario_params(scenario='ScenarioC', N=100, difficulty='Easy', verbose=True):
    if difficulty=='Easy': epsilon=0.01; eta=0.25; zeta = 0
    elif difficulty == 'Hard' : epsilon=0.1; eta=0.25; zeta = 0.7
    else : print('Difficulty should either be Easy or Hard')

    if scenario=='ScenarioA':
        ########### SCENARIO A ###########
        Q = 3
        K = 4

        C            = rng.multinomial(1, np.ones(Q)/Q, size=N) # one hot encoded cluster membership
        # Probability connection matrix
        pi = epsilon * np.ones( (Q,Q))
        np.fill_diagonal(pi, eta)

        # Topic proportions
        topics = np.zeros((Q,Q))
        np.fill_diagonal(topics, range(1,K))

    if scenario=='ScenarioB':
        ########### SCENARIO B ###########
        Q = 2
        K = 3

        C            = rng.multinomial(1, np.ones(Q)/Q, size=N) # one hot encoded cluster membership
        # Probability connection matrix
        pi = eta *  np.ones( (Q,Q))

        # Topic proportions
        topics = np.zeros((Q,Q))
        np.fill_diagonal(topics, range(1,K))

    if scenario=='ScenarioC':
        ########### SCENARIO C ###########
        Q = 4
        K = 3

        C            = rng.multinomial(1, np.ones(Q)/Q, size=N) # one hot encoded cluster membership
        # Probability connection matrix
        pi = epsilon * np.ones( (Q,Q))
        np.fill_diagonal(pi, eta)
        pi[Q-1, Q-2] = eta
        pi[Q-2, Q-1] = eta

        # Topic proportions
        topics = np.zeros((Q,Q), dtype=np.int)
        topics[0,0], topics[2,2] = 1, 1
        topics[1,1], topics[3,3] = 2, 2
    
    topics = (1-zeta) * topics + zeta * np.ones_like(topics) / K
    if verbose:
        print('Cluster proportion', C.sum(0)/N)        
        print('Connection probabilities :\n', pi)
        print('Topics between clusters :\n',topics )

    return Q, K, C.argmax(1), pi, np.int32(topics)


def Simulation_BBC(BBC_root, save_path=None, scenario='ScenarioA', difficulty = 'Easy', N=100,
                   min_words_before_processing = 150, verbose=True):
#number of words in each document before pre-processing

    Q, K, node_cluster, pi, topics = get_scenario_params(scenario, N, difficulty=difficulty, verbose=verbose)
    BBC_articles = []
    for article in ['msgA', 'msgB', 'msgC', 'msgD']:
        with open(os.path.join(BBC_root, article + '.txt'), 'r', encoding='utf-8') as art:
            BBC_articles.append(art.readlines()[0].replace('\n', ' ').split(' '))
    
    # Remove the jumpline
    #BBC_articles = [[word for word in article if '\n' not in word] for article in BBC_articles]
    #print(BBC_articles)
    
    A = np.zeros( (N,N))
    T = np.zeros( (N,N) )
    T[:] = np.nan
    #W = []
    W = np.empty((N,N), dtype='O')
    for i in range(N):
        for j in range(N):
            # node i connects to node j with proba pi[C_{iq},C_{jr]]
            if i != j:
                # No self loop
                A[i,j] = rng.binomial(1, pi[ node_cluster[i], node_cluster[j]])
                if A[i,j]:
                    # If they are connected, a text is generated according to a certain topic
                    topic = topics[node_cluster[i], node_cluster[j]]
                    T[i,j] = int(topic)
                    text = rng.choice(BBC_articles[topic], size = min_words_before_processing)
                    W[i,j] = ' '.join(text)
                    #W.append(' '.join(text))

    # We remove the isolated nodes 
    nodes_to_keep = np.where((A.sum(1) + A.sum(0)) != 0)[0]
    A = A[np.ix_(nodes_to_keep, nodes_to_keep)]
    W = W[np.ix_(nodes_to_keep, nodes_to_keep)]
    T = T[np.ix_(nodes_to_keep, nodes_to_keep)]
    node_cluster = node_cluster[nodes_to_keep]
    if len(nodes_to_keep) != N: 
        print('Number of non connected nodes that had to be dropped : {}'.format(N - len(nodes_to_keep)))   
        
    if verbose:
        # Print the proportions of connections between clusters
        real_connection_proportions = np.zeros((Q,Q))
        clust_size = [ (node_cluster==q).sum() for q in range(Q)]
        for q in range(Q):
            for r in range(Q):
                real_connection_proportions[q,r] = A[np.ix_( node_cluster==q, node_cluster == r)].sum() 
                real_connection_proportions[q,r] /=  (clust_size[q] * clust_size[r])
        print('The proportions of connection between clusters in the dataset is :')
        print(real_connection_proportions)
    
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path,'adjacency.npy'), A)
        np.save(os.path.join(save_path,'topics.npy'), T )
        np.save(os.path.join(save_path,'clusters.npy'), node_cluster)
        np.save(os.path.join(save_path,'text_matrix.npy'), W) 
        W = W[A!=0].tolist()
        with open(os.path.join(save_path, 'texts.txt'), 'w') as file:
            file.write('\n'.join(W))
    
    return Q, K, A, T, W, pi, node_cluster, topics