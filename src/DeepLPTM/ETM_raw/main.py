#/usr/bin/python

from __future__ import print_function

import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
from ETM_raw import data
import scipy.io

from torch import nn, optim
from torch.nn import functional as F

from ETM_raw.etm import ETM
from ETM_raw.utils import nearest_neighbors, get_topic_coherence, get_topic_diversity, arguments


class ETM_algo():    
    def __init__(self, **kwargs):
        # See the utils files for more informations on the arguments
        self.args = arguments(**kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('\n')
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)

        ## get data
        # 1. vocabulary
        self.vocab, train, valid, test = data.get_data(os.path.join(self.args.data_path))
        self.args.vocab_size = len(self.vocab)

        # 1. training data
        self.train_tokens = train['tokens']
        self.train_counts = train['counts']
        self.args.num_docs_train = len(self.train_tokens)

        # 2. dev set
        self.valid_tokens = valid['tokens']
        self.valid_counts = valid['counts']
        self.args.num_docs_valid = len(self.valid_tokens)

        # 3. test data
        self.test_tokens = test['tokens']
        self.test_counts = test['counts']
        self.args.num_docs_test = len(self.test_tokens)
        self.test_1_tokens = test['tokens_1']
        self.test_1_counts = test['counts_1']
        self.args.num_docs_test_1 = len(self.test_1_tokens)
        self.test_2_tokens = test['tokens_2']
        self.test_2_counts = test['counts_2']
        self.args.num_docs_test_2 = len(self.test_2_tokens)

        self.embeddings = None
        if self.args.use_pretrained_emb:
            emb_path = self.args.emb_path
            vect_path = os.path.join(self.args.data_path.split('/')[0], 'embeddings.pkl')   
            vectors = {}
            """
            with open(emb_path, 'rb') as f:
                for l in f:
                    line = l.decode(encoding='utf8').split()
                    word = line[0]
                    if word in self.vocab:
                        vect = np.array(line[1:]).astype(np.float)
                        vectors[word] = vect
            """
            with open(emb_path, 'r') as f:
                file = f.readlines()
                for l in file:
                    line = l.split()
                    word = line[0]
                    if word in self.vocab:
                        vect = np.array(line[1:]).astype(np.float)
                        vectors[word] = vect
            embeddings = np.zeros((self.args.vocab_size, self.args.emb_size))
            words_found = 0
            for i, word in enumerate(self.vocab):
                try: 
                    embeddings[i] = vectors[word]
                    words_found += 1
                except KeyError:
                    embeddings[i] = np.random.normal(scale=0.6, size=(self.args.emb_size, ))
            self.embeddings = torch.from_numpy(embeddings).to(self.device)
            self.args.embeddings_dim = embeddings.size
            del embeddings
        print('=*'*100)
        print('Training an Embedded Topic Model on {} with the following settings: {}'.format(self.args.dataset.upper(), self.args))
        print('=*'*100)

        # define checkpoint
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

        if self.args.mode == 'eval':
            self.args.ckpt = self.args.load_from
        else:
            self.args.ckpt = os.path.join(self.args.save_path, 
                'etm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
                self.args.dataset, self.args.num_topics, self.args.t_hidden_size, self.args.optimizer, self.args.clip, self.args.theta_act, 
                    self.args.lr, self.args.batch_size, self.args.rho_size, self.args.train_embeddings))

        # define model and optimizer
        self.model = ETM(self.args.num_topics, self.args.vocab_size, self.args.t_hidden_size, self.args.rho_size, self.args.emb_size, 
                        self.args.theta_act, self.embeddings, self.args.use_pretrained_emb, self.args.train_embeddings, self.args.enc_drop).to(self.device)

        print('model: {}'.format(self.model))

        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay)
        elif self.args.optimizer == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay)
        elif self.args.optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay)
        elif self.args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay)
        elif self.args.optimizer == 'asgd':
            self.optimizer = optim.ASGD(self.model.parameters(), lr=self.args.lr, t0=0, lambd=0., weight_decay=self.args.wdecay)
        else:
            print('Defaulting to vanilla SGD')
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)

    def train(self, epoch, training=False, aggregate=True, verbose=True):
        self.model.train()
        acc_loss = 0
        acc_kl_theta_loss = 0
        cnt = 0
        indices = torch.randperm(self.args.num_docs_train)
        indices = torch.split(indices, self.args.batch_size)
        for idx, ind in enumerate(indices):
            self.optimizer.zero_grad()
            self.model.zero_grad()
            data_batch = data.get_batch(self.train_tokens, self.train_counts, ind, self.args.vocab_size, self.device)
            sums = data_batch.sum(1).unsqueeze(1)
            if self.args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            recon_loss, kld_theta = self.model(data_batch, normalized_data_batch, training, ind=ind, aggregate=aggregate)
            total_loss = recon_loss + kld_theta
            if self.args.etsbm_training:
                total_loss = total_loss * len(ind) # sum instead of mean in the elbo over the meta docs
            total_loss.backward()

            if self.args.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()

            acc_loss += torch.sum(recon_loss).item()
            acc_kl_theta_loss += torch.sum(kld_theta).item()
            cnt += 1

            if idx % self.args.log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2) 
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
                cur_real_loss = round(cur_loss + cur_kl_theta, 2)
                if verbose:
                    print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                        epoch, idx, len(indices), self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
        
        cur_loss = round(acc_loss / cnt, 2) 
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
        cur_real_loss = round(cur_loss + cur_kl_theta, 2)
        #print('*'*100)
        print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
        #print('*'*100)

        return cur_real_loss

    def visualize(self, m, show_emb=False):
        """
        if not os.path.exists('./results'):
            os.makedirs('./results')
        """
        m.eval()

        queries = ['andrew', 'computer', 'sports', 'religion', 'man', 'love', 
                    'intelligence', 'money', 'politics', 'health', 'people', 'family']
        queries = ['party', 'cancer', 'hole','astronomy', 'duchess']
        ## visualize topics using monte carlo
        with torch.no_grad():
            print('#'*100)
            print('Visualize topics...')
            topics_words = []
            gammas = m.get_beta()
            for k in range(self.args.num_topics):
                gamma = gammas[k]
                top_words = list(gamma.cpu().numpy().argsort()[-self.args.num_words+1:][::-1])
                topic_words = [self.vocab[a] for a in top_words]
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
                        word, nearest_neighbors(word, embeddings, self.vocab)))
                print('#'*100)

    def evaluate(self, m, source, tc=False, td=False, verbose=True):
        """Compute perplexity on document completion.
        """
        m.eval()
        with torch.no_grad():
            if source == 'val':
                indices = torch.split(torch.tensor(range(self.args.num_docs_valid)), self.args.eval_batch_size)
                tokens = self.valid_tokens
                counts = self.valid_counts
            else: 
                indices = torch.split(torch.tensor(range(self.args.num_docs_test)), self.args.eval_batch_size)
                tokens = self.test_tokens
                counts = self.test_counts

            ## get \beta here
            beta = m.get_beta()

            ### do dc and tc here
            acc_loss = 0
            cnt = 0
            indices_1 = torch.split(torch.tensor(range(self.args.num_docs_test_1)), self.args.eval_batch_size)
            for idx, ind in enumerate(indices_1):
                ## get theta from first half of docs
                data_batch_1 = data.get_batch(self.test_1_tokens, self.test_1_counts, ind, self.args.vocab_size, self.device)
                sums_1 = data_batch_1.sum(1).unsqueeze(1)
                if self.args.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1
                theta, _ = m.get_theta(normalized_data_batch_1)

                ## get prediction loss using second half
                data_batch_2 = data.get_batch(self.test_2_tokens, self.test_2_counts, ind, self.args.vocab_size, self.device)
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
            if verbose:
                print('*'*100)
                print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
                print('*'*100)
            if tc or td:
                beta = beta.data.cpu().numpy()
                if tc:
                    if verbose:
                        print('Computing topic coherence...')
                    get_topic_coherence(beta, self.train_tokens, self.vocab)
                if td:
                    if verbose:
                        print('Computing topic diversity...')
                    get_topic_diversity(beta, 25)
            return ppl_dc

    def train_etm(self, training=False, verbose=True, aggregate=True):
        ## train model on data 
        best_epoch = 0
        best_val_ppl = 1e9
        all_val_ppls = []
        Nelbo = []
        if verbose:
            print('\n')
            print('Visualizing model quality before training...')
            self.visualize(self.model)
            print('\n')
        for epoch in range(0, self.args.epochs):
            neg_elbo = self.train(epoch, training=training, aggregate=aggregate, verbose=verbose)
            Nelbo.append(neg_elbo)
            val_ppl = self.evaluate(self.model, 'val', verbose=verbose)
            if val_ppl < best_val_ppl:
                with open(self.args.ckpt, 'wb') as f:
                    torch.save(self.model, f)
                best_epoch = epoch
                best_val_ppl = val_ppl
            else:
                ## check whether to anneal lr
                lr = self.optimizer.param_groups[0]['lr']
                if self.args.anneal_lr and (len(all_val_ppls) > self.args.nonmono and val_ppl > min(all_val_ppls[:-self.args.nonmono]) and self.lr > 1e-5):
                    self.optimizer.param_groups[0]['lr'] /= self.args.lr_factor
            if epoch % self.args.visualize_every == 0 and verbose:
                self.visualize(self.model)
            all_val_ppls.append(val_ppl)
        if self.args.load_after_training:
            with open(self.args.ckpt, 'rb') as f:
                self.model = torch.load(f)
        self.model = self.model.to(self.device)
        val_ppl = self.evaluate(self.model, 'val', verbose=verbose)
        elbo = - np.array(Nelbo)
        return elbo, Nelbo

    def eval(self):
        with open(self.args.ckpt, 'rb') as f:
            self.model = torch.load(f)
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            ## get document completion perplexities
            test_ppl = self.evaluate(self.model, 'test', tc=self.args.tc, td=self.args.td)

            ## get most used topics
            indices = torch.tensor(range(self.args.num_docs_train))
            indices = torch.split(indices, self.args.batch_size)
            thetaAvg = torch.zeros(1, self.args.num_topics).to(self.device)
            thetaWeightedAvg = torch.zeros(1, self.args.num_topics).to(self.device)
            cnt = 0
            for idx, ind in enumerate(indices):
                data_batch = data.get_batch(self.train_tokens, self.train_counts, ind, self.args.vocab_size, self.device)
                sums = data_batch.sum(1).unsqueeze(1)
                cnt += sums.sum(0).squeeze().cpu().numpy()
                if self.args.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch
                theta, _ = self.model.get_theta(normalized_data_batch)
                thetaAvg += theta.sum(0).unsqueeze(0) / self.args.num_docs_train
                weighed_theta = sums * theta
                thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
                if idx % 100 == 0 and idx > 0:
                    print('batch: {}/{}'.format(idx, len(indices)))
            thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
            print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))

            ## show topics
            beta = self.model.get_beta()
            topic_indices = list(np.random.choice(self.args.num_topics, 10)) # 10 random topics
            print('\n')
            for k in range(self.args.num_topics):#topic_indices:
                gamma = beta[k]
                top_words = list(gamma.cpu().numpy().argsort()[-self.args.num_words+1:][::-1])
                topic_words = [self.vocab[a] for a in top_words]
                print('Topic {}: {}'.format(k, topic_words))

            if self.args.train_embeddings:
                ## show etm embeddings 
                try:
                    rho_etm = self.model.rho.weight.cpu()
                except:
                    rho_etm = self.model.rho.cpu()
                queries = ['andrew', 'woman', 'computer', 'sports', 'religion', 'man', 'love', 
                                'intelligence', 'money', 'politics', 'health', 'people', 'family']
                queries = ['party', 'cancer', 'hole','astronomy', 'duchess']

                print('\n')
                print('ETM embeddings...')
                for word in queries:
                    print('word: {} .. etm neighbors: {}'.format(word, nearest_neighbors(word, rho_etm, self.vocab)))
                print('\n')
