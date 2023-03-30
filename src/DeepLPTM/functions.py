import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI
from DeepLPTM.ETM_raw import data
import time
import inspect

def save_results_func(results, save_path):
    f_cuda = lambda key,val : np.save(os.path.join(save_path, key + '.npy'), val.cpu().detach().numpy())
    f_torch = lambda key,val: np.save(os.path.join(save_path, key + '.npy'), val.numpy())
    f_npy   = lambda key,val: np.save(os.path.join(save_path, key + '.npy'), val)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    counter = 0
    for key, value in results.items():
        if 'numpy' in str(type(value)):
            f_npy(key, value)
            counter += 1
        elif 'torch' in str(type(value)):
            if 'torch.cuda' in str(value.type()):
                f_cuda(key, value)
                counter += 1
            else :
                f_torch(key, value)
                counter += 1
    if len(results) != counter:
        print("WARNING, some values have not been saved !")
    else :
        print('The results have been saved.')
        
def load_data(data_path, scenario, args=None):
    """
    Load a graph with the text on the edges
    Arguments :
        - data_path (string): 
            finishing with a backslash
        - scenario  (string): 
            either "ScenarioA", "ScenarioB", "ScenarioC"
        - args      (class of DeepLPM arguments)
    Returns  :
        - adjacency matrix: N x N binary matrix
        - node            : cluster memberships
        - X (np array)    : bag of words
        - T (np array)    : true topics
        - P (int)         : dimension of Z (node latent variable)
        - N (int)         : number of nodes
        - M (int)         : number of edges
        - Q (int)         : number of clusters
        - K (int)         : number of topics
    """
    Q, K = get_true_params(scenario)


    ##################### Load data ########################
    adj      = pd.read_csv(os.path.join(data_path, 'adjacency.csv'),
                           index_col=None, header=None, sep=';').to_numpy()
    labels   = pd.read_csv(os.path.join(data_path, 'clusters.csv'),
                           index_col=None, header=None,sep=';').to_numpy().squeeze()
    X        = pd.read_csv(os.path.join(data_path, 'texts.csv'),
                           index_col=None, header=None, sep='/').to_numpy()
    T        = pd.read_csv(os.path.join(data_path, 'topics.csv'),
                           index_col=None, header=None, sep=';').to_numpy()
    X        = X[adj != 0]
    T        = T[adj != 0]

    #args.hidden2_dim = 3
    N = adj.shape[0]
    M = adj.sum()
    
    P = None
    if args is not None:
        P = 2
        args.num_points   = N # nb nodes
        args.num_clusters = Q
        args.input_dim    = args.num_points
    return adj, labels, X, T, P, N, M, Q, K


def DeepLPM_format(adj, args):
    from scipy import sparse as sp
    from deepLPM_main.preprocessing import preprocess_graph, sparse_to_tuple
    from deepLPM_main import model as Model
    features = np.zeros((adj.shape[0], args.input_dim))
    np.fill_diagonal(features, 1)
    adj      = sp.csr_matrix(adj)
    adj_coo  = adj.tocoo()
    indices  = [adj_coo.row, adj_coo.col] 
    del adj_coo
    features = sp.csr_matrix(features)
    edges    = 0

    ##################### Some preprocessing ########################
    adj_norm = preprocess_graph(adj)  # used to train the encoder
    features = sparse_to_tuple(features.tocoo())
    # adj = sparse_to_tuple(adj)  # original adj
    adj_label = adj + sp.eye(adj.shape[0])  # used to calculate the loss
    adj_label = sparse_to_tuple(adj_label)

    # Create Model
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].astype(float).T),
                                torch.FloatTensor(adj_norm[1].astype(float)),
                                torch.Size(adj_norm[2]))
    # adj = torch.sparse.FloatTensor(torch.LongTensor(adj[0].astype(float).T),
    #                             torch.FloatTensor(adj[1].astype(float)),
    #                             torch.Size(adj[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].astype(float).T),
                                torch.FloatTensor(adj_label[1]),
                                torch.Size(adj_label[2]))
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].astype(float).T),
                                torch.FloatTensor(features[1]), 
                                torch.Size(features[2]))
    edges = torch.Tensor(edges)

    # to GPU
    adj_norm = adj_norm.to(args.device)
    adj_label = adj_label.to(args.device)
    features = features.to(args.device)
    edges = edges.to(args.device)
    return adj, adj_label, adj_norm, features, edges, indices


def one_hot(a, num_classes):
    """
    Given a 1 dimensional array of size N, return a  (N,num_classes) array with binary entries and 1 in the same column for each value
    :param a: 1-d array
    :param num_classes: (int)
    :return: one-hot encoded vector
    """
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])



def check_convergence(x_new, x_old, p='fro', ratio=True, tol=1e-5):
    BREAK = False
    difference = torch.norm(x_new - x_old, p=p)
    if ratio :
        difference /= torch.norm(x_old)
    if  difference < tol:
        print('Convergence has been reached')
        BREAK = True
    return BREAK


def KL_Y(m, log_s, mu_Y, log_cov_Y, Q, K, M, device):
    """
    Arguments : 
        - m (tensor)           : shape (Q**2, K)
        - log_s (tensor)       : shape (Q**2)
        - mu_Y (tensor)        : shape (M, K)
        - log_cov_Y (tensor) : shape (M, K)
    
    Results : 
        - KL( N(mu_Y, diag(sigma_Y^2) ) || N(m,s^2 I_K) )
    """
    Q_2 = Q**2
    assert m.shape == torch.Size([Q_2, K]), "m shape should be : [{:.0f}, {:.0f}]  but is {:}".format(Q_2, K, m.shape)
    assert log_s.shape == torch.Size([Q_2]), "log_s shape should be : [{:.0f}]  but is {:}".format(Q_2, log_s.shape)
    assert mu_Y.shape == torch.Size([M, K]), "mu_Y shape should be : [{:.0f}, {:.0f}]  but is {:}".format(M, K, mu_Y.shape)
    assert log_cov_Y.shape == torch.Size([M, K]), "log_cov_Y shape should be : [{:.0f}, {:.0f}]  but is {:}".format(M, K, log_cov_Y.shape)

    KL = torch.zeros( (M,Q**2)).to(device)
    for qr in range(Q**2):
        log_s_qr = torch.ones(K).to(device) * log_s[qr]
        KL[:,qr] = torch.sum(log_s_qr - log_cov_Y - 1, dim=1)
        KL[:,qr] += torch.sum( log_cov_Y.exp() / log_s_qr.exp(), dim=1)
        KL[:,qr] += torch.sum( (mu_Y - m[qr,:])**2, dim=1) / log_s[qr].exp()
    return 0.5 * KL


def update_tau(tau, mu_phi, log_cov_phi, pi_k, mu_k, log_cov_k, kl_theta, P, M, Q, K, args, use='all', device='cuda'):
    eps = (torch.ones(1) * torch.finfo(torch.double).tiny).to(device)
    
    min_exp = torch.log(torch.ones(1, dtype=torch.double) * torch.finfo(torch.double).tiny).to(device)
    max_exp = torch.log( torch.ones(1, dtype=torch.double) * torch.finfo(torch.double).max ).to(device)
    
    KL = torch.zeros((args.num_points, args.num_clusters), dtype = torch.double)  # N * K
    KL = KL.to(device)
    
    if use in ['all', 'network']:
        for k in range(args.num_clusters):
            log_cov_K = torch.ones_like(log_cov_phi) * log_cov_k[k]
            mu_K = torch.ones((args.num_points, mu_k.shape[1])).to(device) * mu_k[k]
            temp = P * (log_cov_K - log_cov_phi - 1) \
                   + P * torch.exp(log_cov_phi) / torch.exp(log_cov_K) \
                   + torch.norm(mu_K - mu_phi, dim=1, keepdim=True) ** 2 / torch.exp(log_cov_K)
            KL[:, k] = 0.5 * temp.squeeze()
    
    if use in ['all', 'texts']:
        with torch.no_grad():
            kl_theta = kl_theta.type(torch.double)
            for M, (i,j) in enumerate(zip(args.indices[0], args.indices[1])):
                KL[i,:] += tau[j,:] @ kl_theta[M,:].reshape(Q,Q).T
                KL[j,:] += tau[i,:] @ kl_theta[M,:].reshape(Q,Q)
    
    gamma = torch.log(pi_k).unsqueeze(0) - KL 
    gamma -= gamma.max(1).values.unsqueeze(1)
    gamma.clamp_(min=min_exp, max=max_exp)
    gamma = torch.softmax(gamma, 1)
    gamma.clamp_(min=eps, max=1-eps)
    
    return gamma


def ELBO_Loss(tau, pi_q, mu_q, log_cov_q, m, log_s, Q, K, M, A_pred, P, data_batch, normalized_data_batch, adj_label, features, normalized_bows, model, etm, device='cuda'):
    """
    Full ELBO loss with network and texts terms.
    Parameters
    ----------
    tau: tensor with shape (num_nodes, num_clusters)
        The posterior node cluster membership probabilities
    pi_q: tensor with shape (num_clusters,)
        The parameter of the model distribution
    mu_q: tensor with shape (num_clusters, p)
        The mean of latent node positions
    log_cov_q : tensor with shape (num_clusters, p)
        The log of the variance of latent node positions.
        It is homoscedastic, ie Sigma_q = exp(log_cov_q) I_p
    m: tensor with shape (num_edges, num_topics)
        The mean of latent documents positions
    log_s: tensor with shape (num_edges, num_topics)
        The log of the variance of latent documents positions.
        It is homoscedastic, ie Sigma = exp(log_s) I_K
    Q: int
        Number of clusters
    K: int
        Number of topics
    M: int
        Number of edges
    A_pred: Tensor with shape (num nodes, num nodes)
        Posterior probability of connection between each pair of nodes
    P: int
        Dimension of latent node positions space

    Returns
    -------
    -Loss: tensor
        The negative ELBO
    Loss1: tensor
        The reconstruction term of the network
    Loss2: tensor
        The weighted KL of the latent node positions compared to the cluster positions
    Loss3: tensor
        The weighted KL of the text distributions compared to the texts sent between clusters
    Loss4: tensor
        The KL of the node cluster membership probabilities and the proportions of clusters

    """
    N = tau.shape[0]
    # Graph reconstruction loss
    OO = adj_label.to_dense() * (torch.log((A_pred / (1. - A_pred)) + 1e-16)) + torch.log((1. - A_pred) + 1e-16)
    OO = OO.fill_diagonal_(0)
    OO = OO.to(device)
    Loss1 = torch.sum(OO)

    mu_Z, log_cov_Z, _ = model.encoder(features)
    kl_z = KL_Z(mu_q, log_cov_q.squeeze(), mu_Z, log_cov_Z.squeeze(), Q, P, N)
    Loss2 = torch.sum(tau * kl_z)

    """
    mu_Y, log_cov_Y, _ = etm.model.encode(normalized_bows, training) # PB IS HERE !!!!!!
    kl_y = KL_Y(m, log_s.squeeze(), mu_Y, log_cov_Y, Q, K, M)
    tau_ij = (tau[indices[0],:].unsqueeze(-1) * tau[indices[1],:].unsqueeze(1)).reshape(M,Q**2)
    Loss3 = torch.sum( tau_ij  * kl_y )
    """

    recon_loss, kld_theta = etm.model(data_batch, normalized_data_batch, True, range(len(normalized_bows)),
                                         aggregate=True)
    Loss3 = recon_loss
    Loss4 = torch.sum(tau * (torch.log(pi_q.unsqueeze(0)) - torch.log(tau)))

    Loss = Loss1 - Loss2 - Loss3 + Loss4

    return -Loss, Loss1, Loss2, Loss3, Loss4


def get_true_params(sc):
    """
    Input : sc, either 'A', 'B' or 'C'
    Output : true (Q,K) for this scenario
    """
    if sc == 'ScenarioA':
        true_Q, true_K = 3, 4  # number of clusters
    elif sc == 'ScenarioB':
        true_Q, true_K = 2, 3
    elif sc == 'ScenarioC':
        true_Q, true_K = 4, 3
    return true_Q, true_K


def KL_Z(mu, log_sigma, mu_Z, log_cov_Z, Q, P, N):
    """
    Arguments :
        - mu (tensor)          : shape (Q, P)
        - log_sigma (tensor)   : shape (Q)
        - mu_Z (tensor)        : shape (N, P)
        - log_cov_Z (tensor)   : shape (N)

    Results :
        - KL( N(mu_Z, cov_Z^2 I_p) || N(mu,sigma^2 I_p) )
    """
    assert mu.shape == torch.Size([Q, P]), "Mu shape should be : [{:.0f}, {:.0f}]  but is {:}".format(Q, P, mu.shape)
    assert log_sigma.shape == torch.Size([Q]), "log_sigma shape should be : [{:.0f}]  but is {:}".format(Q,
                                                                                                         log_sigma.shape)
    assert mu_Z.shape == torch.Size([N, P]), "mu_Z shape should be : [{:.0f}, {:.0f}]  but is {:}".format(N, P,
                                                                                                          mu_Z.shape)
    assert log_cov_Z.shape == torch.Size([N]), "log_cov_Z shape should be : [{:.0f}]  but is {:}".format(N,
                                                                                                         log_cov_Z.shape)

    P = mu.shape[1]
    Q = mu.shape[0]
    N = mu_Z.shape[0]

    KL = P * (log_sigma - log_cov_Z.unsqueeze(-1) - 1)
    KL += log_sigma.exp().pow(-2).unsqueeze(0) * (
                P * log_cov_Z.exp().pow(2).unsqueeze(-1) + (mu_Z.unsqueeze(1) - mu).pow(2).sum(-1))
    return 0.5 * KL


def KL_Z(tau, mu_q, log_cov_q, pi_q, mu_Z, log_cov_Z, Q, P, N, args, device='cuda'):
    det = 1e-16
    KL = torch.zeros((N, Q), dtype=torch.float32)  # N * K
    KL = KL.to(device)
    for q in range(Q):
        log_cov_Q = torch.ones_like(log_cov_Z) * log_cov_q[q]
        mu_Q = torch.ones((args.num_points, mu_q.shape[1])).to(device) * mu_q[q]
        temp = P * (log_cov_Q - log_cov_Z - 1) \
               + P * torch.exp(log_cov_Z) / torch.exp(log_cov_Q) \
               + torch.norm(mu_Q - mu_Z, dim=1, keepdim=True) ** 2 / torch.exp(log_cov_Q)
        KL[:, q] = 0.5 * temp.squeeze()

    denominator = torch.sum(pi_q.unsqueeze(0) * torch.exp(-KL), axis=1, dtype=torch.float32)
    for q in range(Q):
        tau[:, q] = pi_q[q] * torch.exp(-KL[:, q]) / denominator + det


def update_analytically_Z_params(mu_Z, log_cov_Z, mu_q, log_cov_q, tau, pi_q, P, args, eps=1e-16):
    N_q = torch.sum(tau, axis=0, dtype=torch.float32)

    pi_q = N_q / args.num_points

    for q in range(args.num_clusters):
        tau_q = tau[:, q]  # N * 1
        mu_q[q] = torch.sum(mu_Z * tau_q.unsqueeze(1), axis=0, dtype=torch.float32) / N_q[q]

        diff = P * torch.exp(log_cov_Z) + torch.sum((mu_q[q].unsqueeze(0) - mu_Z) ** 2, axis=1,
                                                    dtype=torch.float32).unsqueeze(1)
        cov_q = torch.sum(tau_q.unsqueeze(1) * diff, axis=0, dtype=torch.float32) / (P * N_q[q])
        log_cov_q[q] = torch.log(cov_q)

    return mu_q, log_cov_q, pi_q


def update_analytically_Y_params(mu_Y, log_cov_Y, m, log_s, tau, indices, args, K):
    tau_ij = (tau[indices[0], :].unsqueeze(2) \
              * tau[indices[1], :].unsqueeze(1)).reshape(len(indices[0]), tau.shape[1] ** 2)
    N_qr = torch.sum(tau_ij, axis=0)

    for qr in range(args.num_clusters ** 2):
        tau_ij_qr = tau_ij[:, qr]  # N * 1
        m[qr] = torch.sum(mu_Y * tau_ij_qr.unsqueeze(1), axis=0, dtype=torch.double) / N_qr[qr]

        diff = torch.exp(log_cov_Y).sum(1).unsqueeze(-1) \
               + torch.sum((m[qr].unsqueeze(0) - mu_Y) ** 2, axis=1, dtype=torch.double).unsqueeze(1)

        cov_qr = torch.sum(tau_ij_qr.unsqueeze(1) * diff, axis=0, dtype=torch.double) / (K * N_qr[qr])
        log_s[qr] = torch.log(cov_qr)

    return m, log_s


def ELBO_Loss_network(gamma, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi, A_pred, adj_label, P, args, device='cuda'):
    # Graph reconstruction loss
    OO = adj_label.to_dense() * (torch.log((A_pred / (1. - A_pred)) + 1e-16)) + torch.log((1. - A_pred) + 1e-16)
    OO = OO.fill_diagonal_(0)
    OO = OO.to(device)
    Loss1 = -torch.sum(OO)

    # KL divergence
    KL = torch.zeros((args.num_points, args.num_clusters))  # N * K
    KL = KL.to(device)
    # old
    # for k in range(args.num_clusters):
    #     for i in range(args.num_points):
    #         KL[i, k] = 0.5 * (P*(log_cov_k[k] - log_cov_phi[i]) - P
    #                           + P*torch.exp(log_cov_phi)[i] / torch.exp(log_cov_k[k])
    #                           + torch.norm(mu_k[k] - mu_phi[i]) ** 2 / torch.exp(log_cov_k[k]))
    # brand new
    for k in range(args.num_clusters):
        log_cov_K = torch.ones_like(log_cov_phi) * log_cov_k[k]
        mu_K = torch.ones((args.num_points, mu_k.shape[1])).to(device) * mu_k[k]
        temp = P * (log_cov_K - log_cov_phi - 1) \
               + P * torch.exp(log_cov_phi) / torch.exp(log_cov_K) \
               + torch.norm(mu_K - mu_phi, dim=1, keepdim=True) ** 2 / torch.exp(log_cov_K)
        KL[:, k] = 0.5 * temp.squeeze()

    Loss2 = torch.sum(gamma * KL)

    Loss3 = torch.sum(gamma * (torch.log(pi_k.unsqueeze(0)) - torch.log(gamma)))

    Loss = Loss1 + Loss2 - Loss3

    return Loss, Loss1, Loss2, -Loss3


def network_step(tau, features, edges, model, args, optimizer, adj_label, P, device='cuda', epoch=0):
    import time
    model.gamma.data = tau.clone()
    #################################### train model ################################################
    t = time.time()
    mu_phi, log_cov_phi, z = model.encoder(features)

    A_pred = model.decoder(z, edges, model.alpha, model.beta)

    # update pi_k, mu_k and log_cov_k
    gamma = model.gamma
    model.update_others(mu_phi.detach().clone(),
                        log_cov_phi.detach().clone(),
                        gamma, args.hidden2_dim)

    # update gamma


    # model.update_gamma(mu_phi.detach().clone(),
    #                   log_cov_phi.detach().clone(),
    #                   pi_k, mu_k, log_cov_k, args.hidden2_dim)

    pi_k = model.pi_k  # pi_k should be a copy of model.pi_k
    log_cov_k = model.log_cov_k
    mu_k = model.mu_k
    gamma = model.gamma
    loss, loss1, loss2, loss3 = ELBO_Loss_network(tau,
                                                  model.pi_k,
                                                  model.mu_k,
                                                  model.log_cov_k,
                                                  mu_phi,
                                                  log_cov_phi,
                                                  A_pred,
                                                  adj_label,
                                                  P,
                                                  args,
                                                  device)


    # calculate of ELBO loss
    optimizer.zero_grad()
    # update of GCN
    loss.backward()
    optimizer.step()

    return loss, loss1, loss2, loss3, A_pred, mu_phi, log_cov_phi


def plot_results_func(store_loss, store_loss1, store_loss2, store_loss3, recon_loss,
                 kld_theta, store_ari_texts, store_ari, mu_phi, clusters, mu_Y, edges,
                 savefig=False, savedir=None, max_len=None, use='all'):
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt
    
    if max_len is None:
        max_len = len(store_loss)
        
    fig = plt.figure( figsize=(15,15), constrained_layout=True )
    gs = GridSpec(4, 3, figure=fig)

    ################################# plots to show results ###################################
    # plot train loss
    #f, ax = plt.subplots(1, figsize=(15, 10))
    #plt.subplot(331)
    #plt.plot(store_loss1.cpu().data.numpy(), color='red')
    #plt.title("Reconstruction loss1")
    ax = fig.add_subplot(gs[0,:])
    ax.plot(store_loss[:max_len] , color='red')
    ax.set_title("Training loss in total")
    ax.grid()

    if use in ['all', 'network']:

        ax = fig.add_subplot(gs[1,0])
        ax.plot(store_loss1[:max_len], color='red')
        ax.set_title("Recons loss")
        ax.grid()

        ax = fig.add_subplot(gs[1,1])
        ax.plot(store_loss2[:max_len], color='red')
        ax.set_title("KL loss2")
        ax.grid()

        ax = fig.add_subplot(gs[1,2])
        ax.plot(store_loss3[:max_len], color='red')
        ax.set_title("Cluster loss3")
        ax.grid()
        
        ax = fig.add_subplot(gs[3,0])
        ax.scatter(mu_phi[:,0],mu_phi[:,1], c=clusters)
        ax.set_title("Latent representation of the nodes")
        
    if use in ['all', 'texts']:
        ax = fig.add_subplot(gs[2,0])
        ax.plot(recon_loss[:max_len], color='blue')
        ax.set_title("Recon loss texts")
        ax.grid()

        ax = fig.add_subplot(gs[2,1])
        ax.plot(kld_theta[:max_len], color='blue')
        ax.set_title("KL texts")
        ax.grid()

        ax = fig.add_subplot(gs[2,2])
        ax.plot(store_ari_texts[:max_len], color='blue')
        ax.set_title("ARI texts")
        ax.grid()

        ax = fig.add_subplot(gs[3,2])
        ax.plot(store_ari[:max_len], color='blue')
        ax.set_title("ARI")
        ax.grid()
    
        ax = fig.add_subplot(gs[3,1])
        ax.scatter(mu_Y[:,0],mu_Y[:,1], c=edges)
        ax.set_title("Latent representation of the texts")
    
    if savefig:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        fig.patch.set_facecolor('white')
        plt.savefig(os.path.join(savedir, 'results.pdf'))


def print_topics(etm):
    with torch.no_grad():
        print('#'*100)
        print('Visualize topics...')
        topics_words = []
        gammas = etm.model.get_beta()
        for k in range(etm.args.num_topics):
            gamma = gammas[k]
            top_words = list(gamma.cpu().numpy().argsort()[-etm.args.num_words+1:][::-1])
            topic_words = [etm.vocab[a] for a in top_words]
            topics_words.append(' '.join(topic_words))
            print('Topic {}: {}'.format(k, topic_words))


def init_dissimilarity(N, Q, theta, A, indices, epsilon=1e-16):
    """Initialize the posterior cluster membershipp probabilities with the dissimilarity algorithm
    Arguments :
        - N (int) : number of nodes
        - Q (int) : number of clusters
        - theta (array) : topic proportions per document
        - A (array) : binary adjacency matrix
        - indices (): non zero coefficient of A
        - epsilon (float) [optional, default=1e-16] :
            minimal value to low bound tau coefficients and upper bound is set to 1 - epsilon
    Returns :
        - tau (array): one hot encoded
    """
    from sklearn_extra.cluster import KMedoids

    X = np.zeros((N, N))
    distance = np.zeros((N, N))

    X[indices[0], indices[1]] = theta.argmax(-1)
    dat = np.hstack((X, X.T))

    B = np.hstack((A, A.T))
    commonNeighbours = B @ B.T
    qui = np.where(commonNeighbours > 0)
    for k in range(qui[0].shape[0]):
        i = qui[0][k]
        j = qui[1][k]
        who = np.where(dat[i, :] * dat[j, :] > 0)[0]
        distance[i, j] = np.sum(dat[i, who] != dat[j, who]) / commonNeighbours[i, j]
    np.fill_diagonal(distance, 0)
    np.fill_diagonal(commonNeighbours, 0)
    meanobs = np.sum(distance) / np.sum(commonNeighbours > 0)
    distance[commonNeighbours == 0] = meanobs
    np.fill_diagonal(distance, 0)

    kmedoids = KMedoids(n_clusters=Q, metric='manhattan', method='pam', max_iter=300, random_state=None)
    labels = kmedoids.fit_predict(distance)
    tau = one_hot(labels, Q)

    return np.clip(tau, epsilon, 1-epsilon)


def list_to_dict(var_list):
    """
    Return a dictionary with the variables names as the keys and the variable values as values.
    """
    results = {}
    for fi in reversed(inspect.stack()):
        # Retrieve the name of the variable
        for var in var_list:
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                # Add the variable into the dictionary
                results[names[0]] = var
    return results