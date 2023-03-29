'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import pandas as pd
from deepLPM_main import args

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    if dataset == 'eveques':
        adjacency = pd.read_csv('data/eveques_new/ResoEvequesClean2021-A.csv',
                                header=0, sep=';').to_numpy()
        ########################### build node features #########################
        if args.use_nodes == True:
            features = pd.read_csv('data/eveques_new/ResoEvequesClean2021-X.csv',
                                   header=0, sep=';').to_numpy()
            for i in range(features.shape[0]):
                if np.isnan(features[i][0]) == True and np.isnan(features[i][1]) == False:
                    features[i][0] = features[i][1]

                elif np.isnan(features[i][0]) == False and np.isnan(features[i][1]) == True:
                    features[i][1] = features[i][0]

                elif np.isnan(features[i][0]) == True and np.isnan(features[i][1]) == True:
                    features[i][0] = features[i][1] = 0

            for i in range(features.shape[0]):
                if features[i][0] == 0 and features[i][1] == 0:
                    features[i][0] = np.mean(features[:, 0])
                    features[i][1] = np.mean(features[:, 1])

            min_f = np.min(features[:, 0:2])
            max_f = np.max(features[:, 0:2])
            features[:, 0:2] = (features[:, 0:2] - min_f) / (max_f - min_f)
        else:
            features = np.zeros((adjacency.shape[0], args.input_dim))
            np.fill_diagonal(features, 1)

        ##################### build edge features (covariates) ###################
        edges_1 = pd.read_csv('data/eveques_new/ResoEvequesClean2021-Ydates.csv',
                                 header=0, sep=';').to_numpy()  # Y_ij^1
        edges_2 = pd.read_csv('data/eveques_new/ResoEvequesClean2021-Yfonctions.csv',
                              header=0, sep=';').to_numpy()  # # Y_ij^3
        edges_3 = pd.read_csv('data/eveques_new/ResoEvequesClean2021-Yregions.csv',
                              header=0, sep=';').to_numpy()  # # Y_ij^2

        edges = np.zeros((args.num_points, args.num_points, args.nb_of_edges))  # N*N*D
        edges[:, :, 0] = edges_1
        edges[:, :, 1] = edges_2
        edges[:, :, 2] = edges_3

    elif dataset == 'cora':
        adjacency = np.loadtxt("data/cora/cora_adj.txt")
        if args.use_nodes == True:
            features = np.loadtxt("data/cora/cora_features.txt")
        else:
            features = np.zeros((adjacency.shape[0], args.input_dim))
            np.fill_diagonal(features, 1)
        edges = np.load("data/cora/cora_edges49.npy")  # N*N*49


    return features, adjacency, edges


