import torch 
import numpy as np
import os
from ETM_raw import data

class arguments():
    def __init__(args_etm,
                 data_path='data/20ng',
                 dataset='20ng',
                 emb_path='data/20ng_embeddings.txt',
                 save_path='./results',
                 batch_size=1000,
                 num_topics=50,
                 rho_size=300,
                 emb_size=300,
                 t_hidden_size=800,
                 theta_act='relu',
                 train_embeddings=1,
                 lr=0.005,
                 lr_factor=4.0,
                 epochs=150,
                 mode='train',
                 optimizer='adam',
                 seed=2019,
                 enc_drop=0.0,
                 clip=0.0,
                 nonmono='10',
                 wdecay=1.2e-6,
                 anneal_lr=0,   
                 bow_norm=1,   
                 num_words=10,
                 log_interval=2,
                 visualize_every=10,
                 eval_batch_size=1000,
                 load_from='',
                 tc=0,
                 td=0,
                 save_res_every=10, 
                 n_batch=10,
                 etsbm_training=False,
                 load_after_training=False,
                 use_pretrained_emb=False,
                 ):
        args_etm.dataset = dataset
        args_etm.data_path = data_path
        args_etm.emb_path = emb_path
        args_etm.save_path = save_path
        args_etm.batch_size = batch_size
        args_etm.num_topics = num_topics
        args_etm.rho_size = rho_size
        args_etm.emb_size = emb_size
        args_etm.t_hidden_size = t_hidden_size
        args_etm.theta_act = theta_act
        args_etm.train_embeddings = train_embeddings
        args_etm.lr = lr
        args_etm.lr_factor = lr_factor
        args_etm.epochs = epochs
        args_etm.mode = mode
        args_etm.optimizer = optimizer
        args_etm.seed = seed
        args_etm.enc_drop = enc_drop
        args_etm.clip = clip
        args_etm.nonmono = nonmono
        args_etm.wdecay = wdecay
        args_etm.anneal_lr = anneal_lr
        args_etm.bow_norm = bow_norm
        args_etm.num_words = num_words
        args_etm.log_interval = log_interval
        args_etm.visualize_every = visualize_every
        args_etm.eval_batch_size = eval_batch_size
        args_etm.load_from = load_from
        args_etm.tc = tc
        args_etm.td = td

        # Added features
        args_etm.load_after_training = load_after_training
        args_etm.save_res_every = save_res_every
        args_etm.etsbm_training=etsbm_training
        args_etm.use_pretrained_emb = use_pretrained_emb

def get_topic_diversity(beta, topk):
    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k,:].argsort()[-topk:][::-1]
        list_w[k,:] = idx
    n_unique = len(np.unique(list_w))
    TD = n_unique / (topk * num_topics)
    print('Topic diveristy is: {}'.format(TD))


def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l].squeeze(0)
            if len(doc) == 1: 
                continue
            else:
                doc = doc.squeeze()
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l].squeeze(0)
        if len(doc) == 1: 
            doc = [doc.squeeze()]
        else:
            doc = doc.squeeze()
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj 


def get_topic_coherence(beta, data, vocab):
    D = len(data) ## number of docs...data is list of documents
    print('D: ', D)
    TC = []
    num_topics = len(beta)
    for k in range(num_topics):
        print('k: {}/{}'.format(k, num_topics))
        top_10 = list(beta[k].argsort()[-11:][::-1])
        top_words = [vocab[a] for a in top_10]
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + ( np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )
                # update tmp: 
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp 
        TC.append(TC_k)
    print('counter: ', counter)
    print('num topics: ', len(TC))
    TC = np.mean(TC) / counter
    print('Topic coherence is: {}'.format(TC))


def nearest_neighbors(word, embeddings, vocab):
    vectors = embeddings.data.cpu().numpy() 
    index = vocab.index(word)
    print('vectors: ', vectors.shape)
    query = vectors[index]
    print('query: ', query.shape)
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:20]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors


######## NEW THINGS

def visualize(m, args_etm, queries = None, show_emb=False):
    if not os.path.exists('./results'):
        os.makedirs('./results')

    m.eval()
    if queries is None:
        queries = ['andrew', 'computer', 'sports', 'religion', 'man', 'love',
                    'intelligence', 'money', 'politics', 'health', 'people', 'family']

    ## visualize topics using monte carlo
    with torch.no_grad():
        print('#'*100)
        print('Visualize topics...')
        topics_words = []
        gammas = m.get_beta()
        for k in range(args_etm.num_topics):
            gamma = gammas[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args_etm.num_words+1:][::-1])
            topic_words = [args_etm.vocab[a] for a in top_words]
            topics_words.append(' '.join(topic_words))
            print('Topic {}: {}'.format(k, topic_words))

        if show_emb:
            ## visualize word embeddings by using V to get nearest neighbors
            print('#'*100)
            print('Visualize word embeddings by using output embedding matrix')
            try:
                embeddings = m.rho.weight  # Vocab_size x E
            except:
                embeddings = m.rho         # Vocab_size x E
            neighbors = []
            for word in queries:
                print('word: {} .. neighbors: {}'.format(
                    word, nearest_neighbors(word, embeddings, args_etm.vocab)))
            print('#'*100)


def evaluate(m, args_etm, tokens, counts, source, device='cuda', tc=False, td=False):
    import math
    """
    Compute perplexity on document completion.
    """
    m.eval()
    with torch.no_grad():
        if source == 'val':
            indices = torch.split(torch.tensor(range(args_etm.num_docs_valid)), args_etm.eval_batch_size)
            tokens = args_etm.valid_tokens
            counts = args_etm.valid_counts
        else:
            indices = torch.split(torch.tensor(range(args_etm.num_docs_test)), args_etm.eval_batch_size)
            tokens = args_etm.test_tokens
            counts = args_etm.test_counts

        ## get \beta here
        beta = m.get_beta()

        ### do dc and tc here
        acc_loss = 0
        cnt = 0
        indices_1 = torch.split(torch.tensor(range(args_etm.num_docs_test_1)), args_etm.eval_batch_size)
        for idx, ind in enumerate(indices_1):
            ## get theta from first half of docs
            data_batch_1 = data.get_batch(args_etm.test_1_tokens, args_etm.test_1_counts, ind, args_etm.vocab_size, device)
            sums_1 = data_batch_1.sum(1).unsqueeze(1)
            if args_etm.bow_norm:
                normalized_data_batch_1 = data_batch_1 / sums_1
            else:
                normalized_data_batch_1 = data_batch_1
            theta, _ = m.get_theta(normalized_data_batch_1)

            ## get prediction loss using second half
            data_batch_2 = data.get_batch(args_etm.test_2_tokens, args_etm.test_2_counts, ind, args_etm.vocab_size, device)
            sums_2 = data_batch_2.sum(1).unsqueeze(1)
            res = torch.mm(theta, beta)
            preds = torch.log(res)
            recon_loss = -(preds * data_batch_2).sum(1)

            loss = recon_loss / sums_2.squeeze()
            loss = loss.mean().item()
            acc_loss += loss
            cnt += 1
        cur_loss = acc_loss / cnt
        ppl_dc = round(math.exp(cur_loss), 1)
        print('*'*100)
        print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
        print('*'*100)
        if tc or td:
            beta = beta.data.cpu().numpy()
            if tc:
                print('Computing topic coherence...')
                get_topic_coherence(beta, args_etm.train_tokens, args_etm.vocab)
            if td:
                print('Computing topic diversity...')
                get_topic_diversity(beta, 25)
        return ppl_dc


def train_etm(deep_etsbm_model,args_etm, aggregate=True):
    ## train model on data
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    Nelbo = []
    print('\n')
    print('Visualizing model quality before training...')
    deep_etsbm_model.visualize(deep_etsbm_model)
    print('\n')
    for epoch in range(0, args_etm.epochs):
        neg_elbo = deep_etsbm_model.train(epoch, aggregate)
        Nelbo.append(neg_elbo)
        val_ppl = deep_etsbm_model.evaluate(deep_etsbm_model, 'val')
        if val_ppl < best_val_ppl:
            with open(args_etm.ckpt, 'wb') as f:
                torch.save(deep_etsbm_model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            lr = deep_etsbm_model.optimizer.param_groups[0]['lr']
            if args_etm.anneal_lr and (len(all_val_ppls) > args_etm.nonmono and val_ppl > min(all_val_ppls[:-args_etm.nonmono]) and deep_etsbm_model.lr > 1e-5):
                deep_etsbm_model.optimizer.param_groups[0]['lr'] /= args_etm.lr_factor
        if epoch % args_etm.visualize_every == 0:
            deep_etsbm_model.visualize(deep_etsbm_model)
        all_val_ppls.append(val_ppl)
    if args_etm.load_after_training:
        with open(args_etm.ckpt, 'rb') as f:
            deep_etsbm_model = torch.load(f)
    deep_etsbm_model = deep_etsbm_model.to(deep_etsbm_model.device)
    val_ppl = deep_etsbm_model.evaluate(deep_etsbm_model, 'val')
    elbo = - np.array(Nelbo)
    return elbo

from torch import optim

def train(deep_etsbm_model, args_etm, epoch, aggregate=True):

    deep_etsbm_model.train()
    acc_loss = 0
    acc_kl_theta_loss = 0
    cnt = 0
    indices = torch.randperm(args_etm.num_docs_train)
    indices = torch.split(indices, args_etm.batch_size)
    for idx, ind in enumerate(indices):
        deep_etsbm_model.optimizer.zero_grad()
        deep_etsbm_model.zero_grad()
        data_batch = data.get_batch(deep_etsbm_model.train_tokens, deep_etsbm_model.train_counts, ind, args_etm.vocab_size, deep_etsbm_model.device)
        sums = data_batch.sum(1).unsqueeze(1)
        if args_etm.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch
        recon_loss, kld_theta = deep_etsbm_model(data_batch, normalized_data_batch,aggregate=aggregate)
        total_loss = recon_loss + kld_theta
        if args_etm.etsbm_training:
            total_loss = total_loss * len(ind) # sum instead of mean in the elbo over the meta docs
        total_loss.backward()

        if args_etm.clip > 0:
            torch.nn.utils.clip_grad_norm_(deep_etsbm_model.parameters(), args_etm.clip)
        deep_etsbm_model.optimizer.step()

        acc_loss += torch.sum(recon_loss).item()
        acc_kl_theta_loss += torch.sum(kld_theta).item()
        cnt += 1

        if idx % args_etm.log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 2)
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
            cur_real_loss = round(cur_loss + cur_kl_theta, 2)

            print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, idx, len(indices), deep_etsbm_model.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))

    cur_loss = round(acc_loss / cnt, 2)
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
    cur_real_loss = round(cur_loss + cur_kl_theta, 2)
    print('*'*100)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, deep_etsbm_model.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
    print('*'*100)

    return cur_loss

def args_etm_init(**kwargs):
    # See the utils files for more informations on the arguments
    args_etm = arguments(**kwargs)

    args_etm.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('\n')
    np.random.seed(args_etm.seed)
    torch.manual_seed(args_etm.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args_etm.seed)

    ## get data
    # 1. vocabulary
    args_etm.vocab, train, valid, test = data.get_data(os.path.join(args_etm.data_path))
    args_etm.vocab_size = len(args_etm.vocab)

    # 1. training data
    args_etm.train_tokens = train['tokens']
    args_etm.train_counts = train['counts']
    args_etm.num_docs_train = len(args_etm.train_tokens)

    # 2. dev set
    args_etm.valid_tokens = valid['tokens']
    args_etm.valid_counts = valid['counts']
    args_etm.num_docs_valid = len(args_etm.valid_tokens)

    # 3. test data
    args_etm.test_tokens = test['tokens']
    args_etm.test_counts = test['counts']
    args_etm.num_docs_test = len(args_etm.test_tokens)
    args_etm.test_1_tokens = test['tokens_1']
    args_etm.test_1_counts = test['counts_1']
    args_etm.num_docs_test_1 = len(args_etm.test_1_tokens)
    args_etm.test_2_tokens = test['tokens_2']
    args_etm.test_2_counts = test['counts_2']
    args_etm.num_docs_test_2 = len(args_etm.test_2_tokens)

    args_etm.embeddings = None
    if not args_etm.train_embeddings:
        emb_path = args_etm.emb_path
        vect_path = os.path.join(args_etm.data_path.split('/')[0], 'embeddings.pkl')
        vectors = {}
        with open(emb_path, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                if word in args_etm.vocab:
                    vect = np.array(line[1:]).astype(np.float)
                    vectors[word] = vect
        embeddings = np.zeros((args_etm.vocab_size, args_etm.emb_size))
        words_found = 0
        for i, word in enumerate(args_etm.vocab):
            try:
                embeddings[i] = vectors[word]
                words_found += 1
            except KeyError:
                embeddings[i] = np.random.normal(scale=0.6, size=(args_etm.emb_size, ))
        embeddings = torch.from_numpy(embeddings).to(args_etm.device)
        args_etm.embeddings_dim = embeddings.size()

    print('=*'*100)
    print('Training an Embedded Topic Model on {} with the following settings: {}'.format(args_etm.dataset.upper(), args_etm))
    print('=*'*100)

    # define checkpoint
    if not os.path.exists(args_etm.save_path):
        os.makedirs(args_etm.save_path)

    if args_etm.mode == 'eval':
        args_etm.ckpt = args_etm.load_from
    else:
        args_etm.ckpt = os.path.join(args_etm.save_path,
            'etm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
            args_etm.dataset, args_etm.num_topics, args_etm.t_hidden_size, args_etm.optimizer, args_etm.clip, args_etm.theta_act,
                args_etm.lr, args_etm.batch_size, args_etm.rho_size, args_etm.train_embeddings))

    if args_etm.optimizer == 'adam':
        args_etm.optimizer = optim.Adam(model.parameters(), lr=args_etm.lr, weight_decay=args_etm.wdecay)
    elif args_etm.optimizer == 'adagrad':
        args_etm.optimizer = optim.Adagrad(model.parameters(), lr=args_etm.lr, weight_decay=args_etm.wdecay)
    elif args_etm.optimizer == 'adadelta':
        args_etm.optimizer = optim.Adadelta(model.parameters(), lr=args_etm.lr, weight_decay=args_etm.wdecay)
    elif args_etm.optimizer == 'rmsprop':
        args_etm.optimizer = optim.RMSprop(model.parameters(), lr=args_etm.lr, weight_decay=args_etm.wdecay)
    elif args_etm.optimizer == 'asgd':
        args_etm.optimizer = optim.ASGD(model.parameters(), lr=args_etm.lr, t0=0, lambd=0., weight_decay=args_etm.wdecay)
    else:
        print('Defaulting to vanilla SGD')
        args_etm.optimizer = optim.SGD(args_etm.model.parameters(), lr=args_etm.lr)