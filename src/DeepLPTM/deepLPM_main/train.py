import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
import numpy as np
import os
import time
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt

from .input_data import load_data
from .preprocessing import *
from . import model
from . import args

# Train on CPU or GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0')  # GPU
# print(torch.cuda.is_available())
# print(device)
# os.environ['CUDA_VISIBLE_DEVICES'] = ""  # CPU
print('Number of clusters:.................'+str(args.num_clusters))


##################### Load data ########################
if args.dataset == 'eveques':
    features, adj, edges = load_data(args.dataset)  # load data
    adj = sp.csr_matrix(adj)
    features = sp.csr_matrix(features)

elif args.dataset == 'cora':
    features, adj, edges = load_data(args.dataset)  # load data
    adj = sp.csr_matrix(adj)
    features = sp.csr_matrix(features)
    labels = np.loadtxt("data/cora/cora_labels.txt")  # class labels

elif args.dataset == 'simuB':
    adj, labels = create_simuB(args.num_points, args.num_clusters, 0.9)  # or 'simuA': create_simuA
    features = np.zeros((adj.shape[0], args.input_dim))
    np.fill_diagonal(features, 1)
    adj = sp.csr_matrix(adj)
    features = sp.csr_matrix(features)
    edges = 0

elif args.dataset == 'simuC':
    adj, labels = create_simuC(args.num_points, args.num_clusters)
    features = np.zeros((adj.shape[0], args.input_dim))
    np.fill_diagonal(features, 1)
    adj = sp.csr_matrix(adj)
    features = sp.csr_matrix(features)
    edges = 0


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
adj_norm = adj_norm.to(device)
adj_label = adj_label.to(device)
features = features.to(device)
edges = edges.to(device)


################################ Model ##################################
# init model and optimizer
model = getattr(model, args.model)(adj_norm)
model.to(device)  # to GPU

model.pretrain(features, adj_label, edges)  # pretraining

optimizer = Adam(model.parameters(), lr=args.learning_rate)  # , weight_decay=0.01

# store loss
store_loss = torch.zeros(args.num_epoch).to(device)
store_loss1 = torch.zeros(args.num_epoch).to(device)
store_loss2 = torch.zeros(args.num_epoch).to(device)
store_loss3 = torch.zeros(args.num_epoch).to(device)
# store_ari = torch.zeros(args.num_epoch).to(device)
store_ari = []

def ELBO_Loss(gamma, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi, A_pred, P):
    # Graph reconstruction loss
    OO = adj_label.to_dense()*(torch.log((A_pred/(1. - A_pred)) + 1e-16)) + torch.log((1. - A_pred) + 1e-16)
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
        temp = P*(log_cov_K-log_cov_phi-1) \
                  + P*torch.exp(log_cov_phi)/torch.exp(log_cov_K) \
                  + torch.norm(mu_K-mu_phi,dim=1,keepdim=True)**2/torch.exp(log_cov_K)
        KL[:, k] = 0.5*temp.squeeze()

    Loss2 = torch.sum(gamma * KL)

    Loss3 = torch.sum(gamma * (torch.log(pi_k.unsqueeze(0)) - torch.log(gamma)))

    Loss = Loss1 + Loss2 - Loss3

    return Loss, Loss1, Loss2, -Loss3


##################### Visualisation of learned embeddings by PCA ####################
from sklearn.decomposition import PCA
def visu():
    if args.dataset == 'eveques':
        labelC = []
        gamma = model.gamma.cpu().data.numpy()
        labels = np.argmax(gamma, axis=1)
        for idx in range(len(labels)):
            if labels[idx] == 0:
                labelC.append('lightblue')
            elif labels[idx] == 1:
                labelC.append('lightgreen')
            elif labels[idx] == 2:
                labelC.append('yellow')
            elif labels[idx] == 3:
                labelC.append('purple')
            elif labels[idx] == 4:
                labelC.append('blue')
            elif labels[idx] == 5:
                labelC.append('orange')
            elif labels[idx] == 6:
                labelC.append('cyan')
            elif labels[idx] == 7:
                labelC.append('red')
            else:
                labelC.append('green')

    pca = PCA(n_components=2, svd_solver='full')
    out = pca.fit_transform(model.encoder.mean.cpu().data.numpy())
    mean = pca.fit_transform(model.mu_k.cpu().data.numpy())
    f, ax = plt.subplots(1, figsize=(15, 10))
    ax.scatter(out[:, 0], out[:, 1], color=labelC)
    # ax.scatter(mean[:, 0], mean[:, 1], color='black', s=50)
    ax.set_title('PCA result of embeddings of DeepLPM (K='+str(args.num_clusters)+')', fontsize=18)
    plt.show()
    # f.savefig("C:/Users/Dingge/Desktop/results/emb_ARVGA.pdf", bbox_inches='tight')


#################################### train model ################################################
begin = time.time()
for epoch in range(args.num_epoch):
    t = time.time()
    mu_phi, log_cov_phi, z = model.encoder(features)

    A_pred = model.decoder(z, edges, model.alpha, model.beta)

    if epoch < 1 or (epoch + 1) % 1 == 0:
        # update pi_k, mu_k and log_cov_k
        gamma = model.gamma
        model.update_others(mu_phi.detach().clone(),
                            log_cov_phi.detach().clone(),
                            gamma, args.hidden2_dim)

        # update gamma
        pi_k = model.pi_k
        log_cov_k = model.log_cov_k
        mu_k = model.mu_k
        model.update_gamma(mu_phi.detach().clone(),
                           log_cov_phi.detach().clone(), 
                           pi_k, mu_k, log_cov_k, args.hidden2_dim)

    pi_k = model.pi_k                    # pi_k should be a copy of model.pi_k
    log_cov_k = model.log_cov_k
    mu_k = model.mu_k
    gamma = model.gamma
    loss, loss1, loss2, loss3 = ELBO_Loss(gamma, pi_k, mu_k, log_cov_k, mu_phi, log_cov_phi, A_pred, args.hidden2_dim)

    if epoch > 1:    
        # calculate of ELBO loss
        optimizer.zero_grad()
        # update of GCN
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 1 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "train_loss1=", "{:.5f}".format(loss1.item()), "train_loss2=", "{:.5f}".format(loss2.item()),
              "train_loss3=", "{:.5f}".format(loss3.item()),
              "time=", "{:.5f}".format(time.time() - t))
        # pred = []
        # for i in range(args.num_points):
        #     if i not in delete:
        #         pred.append(torch.argmax(gamma, axis=1).cpu().numpy()[i])

    # if (epoch + 1) % 2000 == 0:
    #     visu()
        # f, ax = plt.subplots(1, figsize=(10, 15))
        # ax.scatter(model.encoder.mean.cpu().data.numpy()[:, 0], model.encoder.mean.cpu().data.numpy()[:, 1], color=labelC)
        # ax.scatter(model.mu_k.cpu().data.numpy()[:, 0], model.mu_k.cpu().data.numpy()[:, 1], color='black', s=50)
        # ax.set_title("Embeddings after training!")
        # plt.show()

    # if epoch== 3:
    #     temp_pathout = "C:/Users/Dingge/Documents/GitHub/deepLPM/temp/"
    #     import pathlib
    #     PathTemp = pathlib.Path(temp_pathout)
    #     if not PathTemp.exists():
    #         PathTemp.mkdir(parents=True,exist_ok=True)
    #
    #     torch.save(log_cov_k,f"{temp_pathout}/log_cov_k.pt")
    #     torch.save(log_cov_phi, f"{temp_pathout}/log_cov_phi.pt")
    #     torch.save(mu_k, f"{temp_pathout}/mu_k.pt")
    #     torch.save(mu_phi, f"{temp_pathout}/mu_phi.pt")


    store_loss[epoch] = torch.Tensor.item(loss)  # save train loss for visu
    store_loss1[epoch] = torch.Tensor.item(loss1)
    store_loss2[epoch] = torch.Tensor.item(loss2)
    store_loss3[epoch] = torch.Tensor.item(loss3)

    # store_ari[epoch] = torch.tensor(adjusted_rand_index(labels, torch.argmax(gamma, axis=1)))  # save ARI
    if args.dataset == 'eveques':
        print('Unsupervised data without true labels (no ARI) !')
    else:
        store_ari.append(adjusted_rand_score(labels, torch.argmax(gamma, axis=1).cpu().numpy()))

end = time.time()
print('training time ......................:', end-begin)


################################# plots to show results ###################################
# plot train loss
f, ax = plt.subplots(1, figsize=(15, 10))
plt.subplot(231)
plt.plot(store_loss1.cpu().data.numpy(), color='red')
plt.title("Reconstruction loss1")

plt.subplot(232)
plt.plot(store_loss2.cpu().data.numpy(), color='red')
plt.title("KL loss2")

plt.subplot(233)
plt.plot(store_loss3.cpu().data.numpy(), color='red')
plt.title("Cluster loss3")

plt.subplot(212)
plt.plot(store_loss.cpu().data.numpy(), color='red')
plt.title("Training loss in total")

plt.show()

print('Min loss:', torch.min(store_loss), 'K='+str(args.num_clusters), str(args.use_nodes)+str(args.use_edges))
if args.dataset != 'eveques':
    print('Max ARI:', max(store_ari))
print('alpha, beta:', model.alpha, model.beta)

# plot ARI
if args.dataset != 'eveques':
    f, ax = plt.subplots(1, figsize=(15, 10))
    ax.plot(store_ari, color='blue')
    ax.set_title("ARI")
    plt.show()


# ARI with kmeans
# kmeans = KMeans(n_clusters=args.num_clusters).fit(model.encoder.mean.cpu().data.numpy())
# labelk = kmeans.labels_
# print("ARI_embedding:", adjusted_rand_score(labels, labelk))

########################### save data for visualisation in R ##################################
# import csv
# file = open('cora_data_A_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.csv', "w")
# writer = csv.writer(file)
# mean = model.encoder.mean.cpu().data.numpy()
# pred_labels = torch.argmax(gamma, axis=1).cpu().numpy()
# for w in range(args.num_points):
#     writer.writerow([w, mean[w][0],mean[w][1],mean[w][2],mean[w][3],mean[w][4],mean[w][5],mean[w][6],mean[w][7],
#                      mean[w][8],mean[w][9],mean[w][10],mean[w][11],mean[w][12],mean[w][13],mean[w][14],mean[w][15], pred_labels[w]])  # mean[w][8],mean[w][9],mean[w][10],mean[w][11],mean[w][12],mean[w][13],mean[w][14],mean[w][15]
# file.close()
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2, svd_solver='full')
# out = pca.fit_transform(mean)
# np.savetxt('cora_pos_A_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.txt', out)
#
# np.savetxt('cora_cl_A_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.txt', pred_labels)
#
# np.savetxt('cora_mu_k='+str(args.num_clusters)+'_p=16_'+str(args.use_nodes)+str(args.use_edges)+'.txt', model.mu_k.cpu().data.numpy())