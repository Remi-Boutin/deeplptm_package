def deeplptm(adj, W, Q, K, P=2,
             init_type='random',
             save_results=False,
             save_path='deep_lptm_results/',
             initialise_etm=True,
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
             topics=None
             ):
    import os
    from torch.optim import Adam
    import numpy as np
    import torch
    from IPython.display import clear_output
    import pandas as pd
    from sklearn.metrics.cluster import adjusted_rand_score as ARI
    from deepLPM_main import model as Model
    from deepLPM_main import args
    from ETM_raw import data
    from ETM_raw.main import ETM_algo

    from ETM_raw.scripts.data_preprocessing import preprocessing
    from functions import training_graph_vectorization, DeepLPM_format, save_results_func, plot_results_func

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

    # store loss

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