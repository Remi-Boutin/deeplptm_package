### CONFIGS ###
# Data selected from:
# simulated data: 'simuA': LPCM; 'simuB': SBM; 'simuC': circle data;
# real data: 'eveques'; 'cora'.
dataset = 'realdata'
model = 'deepLPM'

if dataset == 'simuB':  # or 'simuA', 'simuC'
    use_nodes = False
    use_edges = False
    nb_of_edges = 0  # covariates dimension D

    num_points = 600  # number of nodes N
    input_dim = 600  # node features dimension (identity matrix in our model)
    hidden1_dim = 64  # hidden layer dimension
    hidden2_dim = 2  # latent dimension P: 16 (best) 2 (visualisation)
    num_clusters = 3  # number of clusters K

    num_epoch = 600  # training epochs
    learning_rate = 2e-3  # 2e-3 or 5e-3 (B: delta<0.4)
    pre_lr = 0.1  # 0.1 (B, C) or 0.2 (A, B: delta<0.4)
    pre_epoch = 100  # pretraining epochs: 100(B: delta<0.6) or 70 (B: delta<0.8, C) or 50

elif dataset == 'eveques':
    use_nodes = False
    use_edges = True
    nb_of_edges = 3  # D

    num_points = 1287  # N
    if use_nodes == True:
        input_dim = 10
    else:
        input_dim = 1287
    hidden1_dim = 64
    hidden2_dim = 16  # P
    num_clusters = 8  # K

    num_epoch = 800
    learning_rate = 2e-3
    pre_lr = 1e-3
    pre_epoch = 100

elif dataset == 'cora':
    use_nodes = True
    use_edges = False
    nb_of_edges = 49  # D

    num_points = 2708  # N
    input_dim = 1433  # dictionary size
    hidden1_dim = 64
    hidden2_dim = 16  # P
    num_clusters = 7  # K (7 classes in cora)

    num_epoch = 2000
    learning_rate = 4e-3
    pre_lr = 0.01  # 0.005
    pre_epoch = 100

elif dataset == 'realdata':  # or 'simuA', 'simuC'
    print('test')
    use_nodes = False
    use_edges = False
    nb_of_edges = 0  # covariates dimension D

    num_points = 149  # number of nodes N
    input_dim = 149  # node features dimension (identity matrix in our model)
    hidden1_dim = 64  # hidden layer dimension
    hidden2_dim = 2  # latent dimension P: 16 (best) 2 (visualisation)
    num_clusters = 5  # number of clusters K

    num_epoch = 600  # training epochs
    learning_rate = 2e-3  # 2e-3 or 5e-3 (B: delta<0.4)
    pre_lr = 0.1  # 0.1 (B, C) or 0.2 (A, B: delta<0.4)
    pre_epoch = 600  # pretraining epochs: 100(B: delta<0.6) or 70 (B: delta<0.8, C) or 50
