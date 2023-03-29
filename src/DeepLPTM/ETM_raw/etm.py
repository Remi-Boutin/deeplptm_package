import torch
import torch.nn.functional as F 
import numpy as np 
import math 

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def KL_Y(m, log_s, mu_Y, log_cov_Y, Q, K, M):
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
    assert log_cov_Y.shape == torch.Size([M, K]), "log_cov_Y shape should be : [{:.0f}, {:.0f}]  but is {:}".format(M, K, log_cov_Z.shape)

    KL = torch.zeros( (M,Q**2)).to(device)
    for qr in range(Q**2):
        log_s_qr = torch.ones(K).to(device) * log_s[qr]
        KL[:,qr] = torch.sum(log_s_qr - log_cov_Y - 1, dim=1)
        KL[:,qr] += torch.sum( log_cov_Y.exp() / log_s_qr.exp(), dim=1)
        KL[:,qr] += torch.sum( (mu_Y - m[qr,:])**2, dim=1) / log_s[qr].exp()
    return 0.5 * KL

class ETM(nn.Module):
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size, emsize, 
                    theta_act, embeddings=None, use_pretrained_emb=False, train_embeddings=True, enc_drop=0.5):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)

        self.theta_act = self.get_activation(theta_act)
        
        ## define the word embedding matrix \rho
        if train_embeddings:
            if use_pretrained_emb:
                self.rho = nn.Linear(rho_size, vocab_size, bias=False)
                self.rho.weight = nn.Parameter(embeddings).to(device)
            else:
                self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            self.rho = embeddings.clone().float().to(device)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)#nn.Parameter(torch.randn(rho_size, num_topics))
    
        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, t_hidden_size), 
                self.theta_act,
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
            )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        std = torch.exp(0.5 * logvar) 
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)

    def encode(self, bows, training=False):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        if training:
            # use my KL
            kl_theta =  KL_Y(self.m, self.log_s, mu_theta, logsigma_theta, self.Q, self.K, bows.shape[0])
        else:
            kl_theta = torch.sum(-0.5 * (1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp() ), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        try:
            logit = self.alphas(self.rho.weight) # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0) ## softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows, return_variational_params=False, training=False):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows, training)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1) 
        if return_variational_params:
            return theta, kld_theta, mu_theta, logsigma_theta
        else:
            return theta, kld_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res+1e-6)
        return preds 

    def forward(self, bows, normalized_bows, training=False, return_variational_params=False, ind=None, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            if return_variational_params:
                theta, kld_theta, mu_theta, logsigma_theta = self.get_theta(normalized_bows, return_variational_params=return_variational_params, training=training)
            else:
                theta, kld_theta = self.get_theta(normalized_bows, training=training)
                
            if training:
                full_kld_theta = kld_theta.clone()
                kld_theta = ((self.tau[self.indices[0][ind],:].reshape(len(ind), self.Q, 1) \
                                        * self.tau[self.indices[1][ind],:].reshape(len(ind),1,self.Q) ).reshape(len(ind), self.Q**2)\
                                      * kld_theta ).sum(1).mean()
        else:
            kld_theta = None

        ## get \beta
        beta = self.get_beta()

        ## get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        
        if aggregate:
            recon_loss = recon_loss.mean()
        else :
            recon_loss = recon_loss.sum()
            kld_theta = kld_theta * normalized_bows.shape[0]
        if return_variational_params:
            return recon_loss, kld_theta, mu_theta, logsigma_theta, full_kld_theta
        else:
            return recon_loss, kld_theta

