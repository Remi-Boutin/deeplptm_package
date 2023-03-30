def deeplptm(adj, W, Q, K, P=2,
             init_type='random',
             save_results=False,
             save_path='deep_lptm_results/',
             initialise_etm=False,
             init_path=None,
             tol=1e-3,
             etm_path=None,
             seed=0,
             max_iter=100,
             preprocess_texts=False,
             max_df=0.9,
             min_df=10,
             etm_init_epochs=80,
             etm_batch_size_init=30,
             use_pretrained_emb=False,
             pretrained_emb_path=None,
             use='all',
             labels=None,
             topics=None,
             return_results=False,
             ):
    """
    Parameters
    ----------
    adj : array
        The binary adjacency matrix
    W : list
        The texts corresponding to the edges
    Q : int
        The number of clusters
    K : int
        The number of topics
    P : int (default = 2)
    init_type : str (default = 'random')
        The choice of the initialisation of the cluster memberships probabilities among
         ['dissimilarity', 'random', 'kmeans', 'deeplpm', 'load']. If 'load' is picked, the init_path argument must be provided
    save_results : bool (default = False)
        Whether to save the results or not
    save_path : str (default = None)
        The path where the results are to be saved
    initialise_etm : bool (default = True)
        Whether to initialise ETM
    init_path : str (default = None),
        The path to the node cluster memberships to use as the initialisation
    tol : float (default = 0.001),
        If the norm of the difference of the node clusters means between two steps is lower than tol,
         the algorithm stops
    etm_path : str (default = None),
        The path of etm initialisation
    seed : int (default = 2023)
        The seed of the initialisation for ETM and the node cluster memberships probabilities
    max_iter : int (default = 100)
        The maximum number of iteration during the optimisation
    preprocess_texts : bool (default = False)
        Whether to preprocess the texts or not
    max_df : float (default = 0.9)
        Maximum document frequency of words in the vocabulary (otherwise, the words are discarded)
    min_df : int (float = 0.0)
        Minimum document frequency of words in the vocabulary (otherwise, the words are discarded)
    etm_init_epochs : int (default = 80)
        Number of iterations during ETM initialisation (used only if initialise_etm = True)
    etm_batch_size_init : int (default = 30)
        Batch size during ETM initialisation (used only if initialise_etm = True)
    use_pretrained_emb : bool (default = False)
        Whether to use pre-trained embeddings. If true, the pretrained_emb_path arguments must be provided
    pretrained_emb_path : str (default = None)
        Path to the pretrained embeddings file
    use : str (default = 'all')
        Which part of the model to use. Choices must be made among ['all', 'texts', 'network"]
    labels : array , optional (default = None)
        The true labels if they are known (for instance if the analysis is made on synthetic data)
    topics : array , optional (default = None)
        The true topics of the documents if they are known (for instance if the analysis is made on synthetic data)
    return_results : bool (default = False)
        Whether the dictionary of the results should be provided or not

    Returns
    -------
    results : dictionary, optional
        If return_results == True, the results are returned
    """
    import os
    from torch.optim import Adam
    import torch
    from IPython.display import clear_output
    from deepLPM_main import model as Model
    from deepLPM_main import args
    from ETM_raw import data
    from ETM_raw.main import ETM_algo
    from ETM_raw.scripts.data_preprocessing import preprocessing
    from DeepLPTM.functions import DeepLPM_format, save_results_func
    assert initialise_etm or etm_path is not None, \
        'ETM should either be initialise or the path of the initialisation should be provided.'

    if etm_path is None:
        etm_path = os.path.join(save_path, 'etm', '')

    if init_path is None:
        init_path = ''

    if use_pretrained_emb:
        assert pretrained_emb_path is not None, 'The path of pretrained embeddings is not provided.'

    N = adj.shape[0]
    M = adj.sum()

    args.etm_init_epochs = etm_init_epochs
    args.K = K
    args.P = P
    args.hidden2_dim = P
    args.M = M
    args.N = N
    args.Q = Q
    args.num_edges = M
    args.num_clusters = Q
    args.tol = tol
    #################### Graph parameters ####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    adj, adj_label, adj_norm, features, edges, indices = DeepLPM_format(adj, args)
    args.indices = indices
    ################################ DeepLPM INIT ##################################
    # init model and optimizer
    model = getattr(Model, args.model)(adj_norm)
    model.to(device)  # to GPU

    # The network modelling is not initialised if the init type is 'random'
    if not init_type == 'random':
        model.pretrain(features, adj_label, edges, verbose=False)  # pretraining

    optimizer = Adam(model.parameters(), lr=args.learning_rate)  # , weight_decay=0.01
    model = model.to(device)

    args.num_epoch = max_iter
    ################################ ETM INIT ##################################
    my_etm = None
    if use in ['all', 'texts']:
        # Creation of the folder where the results are to be saved
        if not os.path.exists(
                os.path.join(etm_path, 'etm_init_pretrained_{}.pt'.format(int(use_pretrained_emb)))):
            if not os.path.exists(etm_path):
                os.makedirs(etm_path)

        if preprocess_texts:
            preprocessing(W, path_save=etm_path, max_df=max_df, min_df=min_df, prop_Tr=1, vaSize = 0)

        if initialise_etm:
            print('Initialise ETM {}'.format(initialise_etm))
            assert os.path.exists(os.path.join(etm_path, 'vocab.pkl')), \
                "Texts have not been pre-processed, impossible to pretrain ETM."


            #### ETM TRAINING ####
            my_etm = ETM_algo(data_path=etm_path,
                              dataset='Texts',
                              seed=seed,
                              enc_drop=0,
                              use_pretrained_emb=use_pretrained_emb,
                              emb_path=pretrained_emb_path,
                              save_path=etm_path,
                              batch_size=etm_batch_size_init,
                              epochs=etm_init_epochs,
                              num_topics=K)

            my_etm.model.float()
            my_etm.train_etm()
            torch.save(my_etm,
                       os.path.join(etm_path, 'etm_init_pretrained_{}.pt'.format(int(use_pretrained_emb))))
        else:
            my_etm = torch.load(
                os.path.join(etm_path, 'etm_init_pretrained_{}.pt'.format(int(use_pretrained_emb))))

        bows = data.get_batch(my_etm.train_tokens, my_etm.train_counts, range(my_etm.train_tokens.shape[0]),
                              my_etm.args.vocab_size, my_etm.device)
        sums = bows.sum(1).unsqueeze(1)
        normalized_bows = bows / sums
        clear_output()

    results = training_graph_vectorization(adj_label,
                                           features,
                                           edges,
                                           optimizer,
                                           my_etm,
                                           model,
                                           args,
                                           adj=adj.toarray(),
                                           epochs=args.num_epoch,
                                           tol=args.tol,
                                           use=use,
                                           ratio=False,
                                           labels=labels,
                                           topics=topics,
                                           init=init_type,
                                           init_path=init_path,
                                           full_batch=False,
                                           device=device,
                                          )
    # Saving results
    if save_results:
        save_results_func(results, save_path=save_path)
        torch.save(my_etm, os.path.join(save_path, 'etm_after_training.pt'))
        torch.save(model.state_dict(), os.path.join(save_path, 'DeepLPM_after_training.pt'))

    if return_results:
        return results


def training_graph_vectorization(adj_label, features, edges, optimizer, etm, deeplpm, args,
                                 labels=None, topics=None,
                                 epochs=50, tol=1e-2, adj=None, ratio=False, use='all',
                                 init='random', init_path=None, full_batch=False, device=None):
    """
    Function to train our model, given an already initialized ETM instance and DeepLPM instance.
    Arguments:
        - adj_label : provided by the init of DeepLPM
        - features  : provided by the init of DeepLPM
        - edges     : provided by the init of DeepLPM
        - optimizer : provided by the init of DeepLPM
        - etm       : instance of class ETM
                     already initialised
        - deeplpm   : instance of class DeepLPM
                    already initialised
        - args      : DeepLPM args
        - epochs    : int
                    number maximum of epochs
        - tol       : float
                    tolerance under which the algorithm is assumed to have converged (default=1e-5)
        - adj       : (opt) array
                    binary adjacency matrix for the dissimilarity init only
        - use       : str
                    either 'all', 'texts' or 'network'.
                        'texts' corresponds to training the model C_i, C_j --> Y_ij --> W_ij
                        'network' corresponds to training the DeepLPM model alone
        - labels    : (opt) array
                    array of the true nodes labels (default=None)
        - T         : (opt) array
                    array of the true edges labels (default=None)
        - init      : string
                    either 'random', 'true_init' (requires to provide labels) (default = 'random'),
                    'dissimilarity' (requires to provide adjacency matrix as array)
                    'load' requires to provide 'init_path' or 'deeplpm'

        - init_path : (opt) str
                    if init = 'load', path of the labels
        - full_batch: bool
                    Whether to train ETM using the full batch (more smooth) or mini-batches (faster)
                    using a batch-size set by etm.args.batch_size (default = False)
        - device    : (opt) str
                    Either 'cuda' or 'cpu' (default=None and 'cuda' will be used if it is available, otherwise the 'cpu' will be used)
    Returns:
        - results   : dict with the following keys:
            - tau         : array
                probabilities of node cluster memberships
            - theta       : array
                estimated topic proportions of each document
            - mu_Y        : array
                posterior mean of the edge positions
            - log_cov_Y   : array
                posterior log covariance of the edge positions
            - mu_Z        : array
                posterior mean of the node positions
            - log_cov_Z   : array
                posterior log covariance of the node positions
            - elbo        : array
                elbo values at each iteration
            - etm         : etm Model
                estimated etm part after Deep-LPTM training
            - deeplpm     : deeplpm Model
                estimated deeplpm part after Deep-LPTM training

    """
    import torch
    import time
    import numpy as np
    from DeepLPTM.ETM_raw import data
    from DeepLPTM.functions import init_dissimilarity, one_hot, network_step, check_convergence
    from DeepLPTM.functions import update_analytically_Y_params, update_tau, ELBO_Loss_network
    from sklearn.metrics import adjusted_rand_score as ARI
    assert init in ['random', 'dissimilarity', 'load', 'deeplpm'],\
        "init should be either 'random', 'load' 'dissimilarity', 'deeplpm' "
    assert use in ['all', 'texts', 'network'], "all should either be one of 'all', 'texts' or 'network'"

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use in ['all', 'texts']:
        bows = data.get_batch(etm.train_tokens, etm.train_counts, range(etm.train_tokens.shape[0]),
                              etm.args.vocab_size, etm.device)
        sums = bows.sum(1).unsqueeze(1)
        normalized_bows = bows / sums

        ################################ INIT OF THE ETM PRIOR MIXTURE ON Y ##################################
        MV = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(args.K),
                                                                        covariance_matrix=0.5 * torch.eye(args.K))
        etm.model.m = MV.sample((args.num_clusters ** 2,)).to(device)
        etm.model.log_s = torch.FloatTensor(args.num_clusters ** 2, 1).fill_(0.1).to(device).squeeze()
        etm.model.Q = args.num_clusters
        etm.model.K = args.K
        etm.model.indices = args.indices

        etm.args.epochs = 1
        if full_batch:
            etm.args.batch_size = args.num_points
    # store loss
    network_elbo = torch.zeros(args.num_epoch)  # .to(device)
    recon_network = torch.zeros(args.num_epoch)  # .to(device)
    KL_network = torch.zeros(args.num_epoch)  # .to(device)
    clust_loss = torch.zeros(args.num_epoch)  # .to(device)
    texts_elbo_list = torch.zeros(args.num_epoch)  # .to(device)
    recon_loss_text = torch.zeros(args.num_epoch)  # .to(device)
    store_ari = torch.zeros(args.num_epoch)  # .to(device)
    kl_text = torch.zeros(args.num_epoch)  # .to(device)
    store_ari_texts = torch.zeros(args.num_epoch)  # .to(device)
    total_loss = torch.zeros(args.num_epoch)

    ################# Initialisation of parameters #################
    if init == 'random':
        from torch.distributions import Dirichlet
        dirichlet = Dirichlet(torch.ones(args.num_clusters))
        tau = dirichlet.sample((args.num_points,)).to(device)

    elif init == 'dissimilarity':
        with torch.no_grad():
            theta, _ = etm.model.get_theta(normalized_bows)
            theta = theta.detach().cpu().numpy()
        tau = init_dissimilarity(args.num_points, args.num_clusters, theta, adj, args.indices, epsilon=1e-16)
        tau = torch.Tensor(tau).to(device)

    elif init == 'load':
        tau = np.load(init_path)
        tau = one_hot(tau, args.num_clusters)
        tau = torch.Tensor(np.clip(tau, 1e-16, 1 - 1e-16)).to(device)

    elif init == 'deeplpm':
        tau = deeplpm.gamma.clone()

    if labels is not None:
        print('ARI of initialisation : {:.2f}'.format(ARI(tau.argmax(1).detach().cpu(), labels)))

    texts_elbo, recon_loss, kl_y, network_loss, loss1, loss2, loss3 = 0, 0, 0, 0, 0, 0, 0
    deeplpm.gamma.data = tau.clone()
    if use in ['all', 'texts']:
        etm.model.tau = tau.clone()
    mu_phi = torch.zeros(args.num_points, args.P)
    log_cov_phi = torch.zeros(args.num_points)
    pi_q = tau.sum(0) / args.num_points
    tau_old = tau.clone()
    mu_k_old = deeplpm.mu_k.clone()
    mu_Y = torch.zeros(args.num_edges, args.K)
    log_cov_Y = torch.zeros(args.num_edges, args.K)
    etm.model.rho.requires_grad_ = False
    etm.model.alphas.requires_grad_ = False

    ##########################
    ####### PARAMETERS ESTIMATION
    ##########################
    for epoch in range(epochs):
        t = time.time()

        ################# NETWORK STEP #################
        if use in ['all', 'network']:
            loss_network, _, _, _, A_pred, mu_phi, log_cov_phi = network_step(tau, features,
                                                                              edges, deeplpm, args,
                                                                              optimizer, adj_label, args.P,
                                                                              device, epoch)

        ################# TEXTS STEP #################
        if use in ['all', 'texts']:
            etm.train_etm(training=True, verbose=False)

            with torch.no_grad():
                # Creation of the KL matrix for nothing (time consuming!)
                mu_Y, log_cov_Y, _ = etm.model.encode(normalized_bows)
                etm.model.m, etm.model.log_s = update_analytically_Y_params(mu_Y, log_cov_Y,
                                                                            etm.model.m,
                                                                            etm.model.log_s,
                                                                            tau, args.indices,
                                                                            args, args.K)

        ################# TAU AND PI UPDATE #################
        pi_q = tau.sum(0) / args.num_points

        with torch.no_grad():
            # kl_y = KL_Y(etm.model.m, etm.model.log_s, mu_Y, log_cov_Y, Q, K, M, args.device)
            if use in ['all', 'texts']:
                mu_Y, _, kl_y = etm.model.encode(normalized_bows, training=True)

            tau = update_tau(tau.type(torch.double), mu_phi, log_cov_phi,
                             pi_q, deeplpm.mu_k, deeplpm.log_cov_k,
                             kl_y, args.P, args.num_edges, args.num_clusters, args.K, args, use=use, device=device)

            if use in ['all', 'network']:
                deeplpm.gamma.data = tau.clone()
            if use in ['all', 'texts']:
                etm.model.tau = tau.clone()
        ################# ELBO COMPUTATION #################
        if use in ['all', 'network']:
            ### GET ELBO TERMS ###
            with torch.no_grad():
                network_loss, loss1, loss2, loss3 = ELBO_Loss_network(tau, pi_q, deeplpm.mu_k,
                                                                      deeplpm.log_cov_k,
                                                                      mu_phi, log_cov_phi, A_pred, adj_label,
                                                                      args.P, args, device)
        if use in ['all', 'texts']:
            with torch.no_grad():
                recon_loss, kl_y = etm.model(bows, normalized_bows, training=True, ind=range(args.num_edges))
                # Compute the entropy like term w.r.t tau in any case
                loss3 = torch.sum(tau * (torch.log(pi_q.unsqueeze(0)) - torch.log(tau)))

        ################# STORE VALUES #################
        total_loss[epoch] = loss1 + loss2 - loss3 + args.num_edges * (recon_loss + kl_y)

        if use in ['all', 'network']:
            network_elbo[epoch] = loss1 + loss2 - loss3
            recon_network[epoch] = loss1
            KL_network[epoch] = loss2
            clust_loss[epoch] = - loss3

        if use in ['all', 'texts']:
            recon_loss_text[epoch] = recon_loss * args.num_edges
            kl_text[epoch] = kl_y * args.num_edges
            texts_elbo_list[epoch] = args.num_edges * (recon_loss + kl_y) - loss3

        if topics is not None:
            store_ari_texts[epoch] = ARI(topics, mu_Y.detach().cpu().numpy().argmax(1))
            print("ARI texts=", '{:.3f}'.format(store_ari_texts[epoch]))

        if labels is not None:
            store_ari[epoch] = ARI(tau.detach().cpu().argmax(1), labels)
            print("ARI= {:.3f}".format(store_ari[epoch]))

        print("Epoch:", '%04d' % (epoch + 1), "network_loss =", "{:.5f}".format(network_elbo[epoch]),
              "train_loss1=", "{:.5f}".format(recon_network[epoch]), "train_loss2=",
              "{:.5f}".format(KL_network[epoch]),
              "train_loss3=", "{:.5f}".format(clust_loss[epoch]),
              "KL texts = {:.3f}".format(kl_y),
              "recon loss texts = {:.3f}".format(recon_loss),
              "time=", "{:.2f} sec".format(time.time() - t))

        #################  CHECK CONVERGENCE #################
        if check_convergence(deeplpm.mu_k, mu_k_old, ratio=ratio, tol=tol):
            break
        # tau_old = tau.clone()
        mu_k_old = deeplpm.mu_k.clone()
    theta, _ = etm.model.get_theta(normalized_bows)

    results = {'tau': tau.detach().cpu().numpy(),
               'theta': theta.detach().cpu().numpy(),
               'mu_Y': mu_Y.detach().cpu().numpy(),
               'log_cov_Y': log_cov_Y.detach().cpu().numpy(),
               'mu_Z': mu_phi.detach().cpu().numpy(),
               'log_cov_Z': log_cov_phi.detach().cpu().numpy(),
               'elbo': - total_loss[:epoch].detach().cpu().numpy(),
               'etm': etm,
               'deeplpm': deeplpm,
               }
    if topics is not None:
        results['text_ari'] = store_ari_texts[:epoch].detach().cpu().numpy()
    if labels is not None:
        results['node_ari'] = store_ari[:epoch].detach().cpu().numpy()
    return results