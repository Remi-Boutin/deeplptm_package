### CONFIGS ###
# Data selected from:
# simulated data: 'simuA': LPCM; 'simuB': SBM; 'simuC': circle data;
# real data: 'eveques'; 'cora'.
dataset = 'bbc'
model = 'deepLPM'

use_nodes = False
use_edges = False
nb_of_edges = 0  # covariates dimension D

num_points = 100  # number of nodes N
input_dim = 10  # node features dimension (identity matrix in our model)

hidden1_dim = 64  # hidden layer dimension
hidden2_dim = 2  # latent dimension P: 16 (best) 2 (visualisation)
num_clusters = 3  # number of clusters K

num_epoch = 600  # training epochs
learning_rate = 2e-3  # 2e-3 or 5e-3 (B: delta<0.4)
pre_lr = 0.1  # 0.1 (B, C) or 0.2 (A, B: delta<0.4)
pre_epoch = 100  # pretraining epochs: 100(B: delta<0.6) or 70 (B: delta<0.8, C) or 50
