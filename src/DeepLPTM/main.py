
if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-K', type=int, help='Number of topics', required=True)
    parser.add_argument('-Q', type=int, help='Number of node clusters', required=True)
    parser.add_argument('-P', type=int, help='Number of node clusters', default=2)
    parser.add_argument('--data_path', help='path to the binary adjacency matrix (in a adjacency.csv file),'
                                            ' the texts (in a texts.csv file)', required=True)
    parser.add_argument('--save_results', type=bool, help='Whether to save the results or not', required=True)
    parser.add_argument('--save_path', type=str, help='path where the results should be saved', default=None)
    parser.add_argument('--clusters_provided', type=bool,
                        help='whether the clusters are provided (in clusters.csv in the data_path folder)', default=False)
    parser.add_argument('--topics_provided', type=bool,
                        help='whether the topics are provided (in topics.csv file in the data_path folder)', default=False)
    parser.add_argument('--init_type', type=str, help='type of initialisation for tau',
                        choices=['dissimilarity', 'random', 'kmeans'], default='dissimilarity', required=True)
    parser.add_argument('--init_path', type=str,
                        help="Path to the node cluster memberships, required and useful only if init_type=='load'")
    parser.add_argument('--max_iter', type=int,
                        help='Maximum number of iterations if the convergence has not been reached', default=50)
    parser.add_argument('--tol', type=float, help='if the norm of the difference of two consecutives'
                                                  ' node cluster positions is lower than the tolerance,'
                                                  ' the algorithm stops.', default=1e-3)
    parser.add_argument('--initialise_etm', type=bool,
                        help='Whether to train a new instance of ETM or not', default=False, required=True)
    parser.add_argument('--etm_init_epochs', type=int,
                        help='Number of epochs to train the new ETM instance', default=80)
    parser.add_argument('--seed', type=int, help='seed', default=2023)
    parser.add_argument('--preprocess_texts', type=bool, help='whether to preprocess texts or not', default=True)
    parser.add_argument('--max_df', type=float,
                        help='maximum document frequency of words kept in vocabulary', default=1.0)
    parser.add_argument('--min_df', type=float,
                        help='minimum document frequency of words kept in vocabulary', default=0.0)
    parser.add_argument('--etm_batch_size_init', type=int, help='Batch size during topic modelling init', default=30)
    parser.add_argument('--use_pretrained_emb', type=bool,
                        help='Use pre trained embeddings (should be pre-trained before hand)', default=False)
    parser.add_argument('--pretrained_emb_path', type=str,
                        help=" Path to the pretrained embeddings, required and useful if use_pretrained_emb == 'True'")
    parser.add_argument('--use', type=str, choices=['all', 'texts', 'network'],
                        help='Which part of the model to use', default='all')
    meta_args, _ = parser.parse_known_args()

    print('Number of clusters Q = {}'.format(meta_args.Q),
          'number of topics K = {}'.format(meta_args.K),
          'init type : {}'.format(meta_args.init_type),
          'save results : {}'.format(meta_args.save_results),
          'initialise ETM : {}'.format(meta_args.initialise_etm))

    from src.DeepLPTM.model import deeplptm
    import pandas as pd

    # Load the data in the data_path folder
    A = pd.read_csv(os.path.join(meta_args.data_path, 'adjacency.csv'), index_col=None, header=None, sep=';').to_numpy()
    W = pd.read_csv(os.path.join(meta_args.data_path, 'texts.csv'), index_col=None, header=None, sep='/').to_numpy()
    W = W[A != 0].tolist()

    if meta_args.topics_provided:
        T = pd.read_csv(os.path.join(meta_args.data_path, 'topics.csv'), index_col=None, header=None, sep=';').to_numpy()
        topics = T[A != 0]
    else:
        topics = None

    if meta_args.clusters_provided:
        node_clusters = pd.read_csv(os.path.join(meta_args.data_path, 'clusters.csv'),
                                    index_col=None, header=None, sep=';').to_numpy().squeeze()
    else:
        node_clusters = None

    # Path to save ETM files and init
    etm_path = os.path.join(meta_args.data_path, 'etm/')

    # Path where to save the results (if none is provided)
    if meta_args.save_path is None:
        meta_args.save_path = os.path.join(meta_args.data_path, 'results_deeplptm/')

    # Fit deeplptm model
    res = deeplptm(A,
                   W,
                   meta_args.Q,
                   meta_args.K,
                   P=meta_args.P,
                   init_type=meta_args.init_type,
                   init_path=meta_args.init_path,
                   save_results=meta_args.save_results,
                   labels=node_clusters,
                   topics=topics,
                   max_iter=meta_args.max_iter,
                   tol=meta_args.tol,
                   save_path=meta_args.save_path,
                   etm_path=etm_path,
                   initialise_etm=meta_args.initialise_etm,
                   etm_init_epochs=meta_args.etm_init_epochs,
                   etm_batch_size_init=meta_args.etm_batch_size_init,
                   pretrained_emb_path=meta_args.pretrained_emb_path,
                   seed=meta_args.seed,
                   preprocess_texts=meta_args.preprocess_texts,
                   max_df=meta_args.max_df,
                   min_df=meta_args.min_df,
                   use=meta_args.use
                   )
