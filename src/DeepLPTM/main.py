import distutils.util
import argparse


def add_bool_arg(parser, name, required = False, default=False, help=None):
    # To obtain boolean features using --name (True) or --no-name (False) but
    # with both arguments mutually exclusive (they cannot be provided at the same time)
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument('--' + name, dest=name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=name, action='store_false', help='--' + name + ' is set to False')
    parser.set_defaults(**{name: default})


if __name__ == '__main__':
    import os
    import sys
    sys.path.append('C:/Users/remib/Documents/2022/deeplptm_package/src/')
    print('haha')

    parser = argparse.ArgumentParser()
    parser.add_argument('-K', type=int, help='Number of topics', required=True)
    parser.add_argument('-Q', type=int, help='Number of node clusters', required=True)
    parser.add_argument('-P', type=int, help='Number of node clusters', default=2)
    parser.add_argument('--data_path', help='path to the binary adjacency matrix (in a adjacency.csv file),'
                                            ' the texts (in a texts.csv file)', required=True)
    add_bool_arg(parser, 'save_results', default=True, required=True)
    parser.add_argument('--save_path', type=str, help='path where the results should be saved', default=None)
    add_bool_arg(parser, 'clusters_provided', default=False,
                 help='whether the clusters are provided (in clusters.csv in the data_path folder)')
    add_bool_arg(parser, 'topics_provided', default=False,
                 help='whether the topics are provided (in topics.csv file in the data_path folder)')
    parser.add_argument('--init_type', type=str, help='type of initialisation for tau',
                        choices=['dissimilarity', 'random', 'deeplpm', 'load'], default='dissimilarity', required=True)
    parser.add_argument('--init_path', type=str,
                        help="Path to the node cluster memberships, required and useful only if init_type=='load'")
    parser.add_argument('--max_iter', type=int,
                        help='Maximum number of iterations if the convergence has not been reached', default=50)
    parser.add_argument('--tol', type=float, help='if the norm of the difference of two consecutives'
                                                  ' node cluster positions is lower than the tolerance,'
                                                  ' the algorithm stops.', default=1e-3)

    add_bool_arg(parser, 'initialise_etm', default=False, help='Train a new instance of ETM')
    parser.add_argument('--etm_init_epochs', type=int,
                        help='Number of epochs to train the new ETM instance', default=80)
    parser.add_argument('--seed', type=int, help='seed', default=2023)
    add_bool_arg(parser, 'preprocess_texts', default=True, help='preprocess the texts to initialise ETM')
    parser.add_argument('--max_df', type=float,
                        help='maximum document frequency of words kept in vocabulary', default=1.0)
    parser.add_argument('--min_df', type=float,
                        help='minimum document frequency of words kept in vocabulary', default=0.0)
    parser.add_argument('--etm_batch_size_init', type=int, help='Batch size during topic modelling init', default=30)
    add_bool_arg(parser, 'pretrained_emb', default=False,
                 help='Use pre trained embeddings (should be pre-trained before hand)')
    parser.add_argument('--pretrained_emb_path', type=str,
                        help=" Path to the pretrained embeddings, required and useful if use_pretrained_emb == 'True'")
    parser.add_argument('--use', type=str, choices=['all'], #ToDo : add the following parts 'texts', 'network',
                        help='Which part of the model to use', default='all')

    meta_args = parser.parse_args()

    print('Number of clusters Q = {}'.format(meta_args.Q),
          'number of topics K = {}'.format(meta_args.K),
          'init type : {}'.format(meta_args.init_type),
          'save results : {}'.format(meta_args.save_results),
          'initialise ETM : {}'.format(meta_args.initialise_etm))

    from DeepLPTM.model import deeplptm

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
