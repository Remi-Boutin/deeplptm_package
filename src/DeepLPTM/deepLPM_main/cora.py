import pandas as pd
import numpy as np

# load data
raw_data = pd.read_csv('data/cora/cora.content',sep = '\t',header = None)
num = raw_data.shape[0]  # N = 2708

papers = pd.read_csv('data/cora/CoRA_Raw/papers_dataset.txt',sep = ';',header = None)  # raw cora
inside = []  # papers can be found in raw cora
for i in list(map.keys()):
    if i in papers.iloc[:,0].to_list():
        inside.append(i)

# change index to [0,2707]
a = list(raw_data.index)
b = list(raw_data[0])
c = zip(b,a)
map = dict(c)

# obtain node features
features = raw_data.iloc[:,1:-1]
# D = 1433
print(features.shape)
features = features.to_numpy()

# obtain one-hot labels
labels = pd.get_dummies(raw_data[1434])
print(labels.head(3))
labels = labels.to_numpy()
list_labels = np.argmax(labels, axis=1)

# load cite infos
raw_data_cites = pd.read_csv('data/cora/cora.cites',sep = '\t',header = None)
data_cites = raw_data_cites.to_numpy()

# find papers who cite paper 4330
cites = list(np.where(raw_data_cites[0]==4330)[0])
papers = []
for index in cites:
    papers.append(data_cites[index][1])
nodes = []
for p in papers:
    nodes.append(map[p])
cls[nodes]
labels[nodes]

matrix = np.zeros((num,num))
edges = np.zeros((num,num,49))  # 3 type of citations: same class, diff class and no infos (most of situations).
# edges = np.zeros((num,num))  # 3 type of citations: same class, diff class and no infos (most of situations).
# edges[:, :, 2] = 1

# number of same words / number of words in total
# for i in range(num):
#     for j in range(i, num):
#         somme = features[i] + features[j]
#         num_words = len(np.where(somme > 0)[0])
#         edges[i, j, 0] = edges[j, i, 0] = len(np.where(somme == 2)[0]) / num_words  # normalisation

# class info
for i in range(num):
    for j in range(i, num):
        mat_label = np.zeros((7, 7))
        mat_label[list_labels[i], list_labels[j]] = 1
        edges[i, j, :] = edges[j, i, :] = mat_label.reshape(-1)
np.save("data/cora/cora_edges49", edges)

dat = np.load("data/cora/cora_edges49.npy")

# create adj
for i,j in zip(raw_data_cites[0],raw_data_cites[1]):
    x = map[i] ; y = map[j]  # index to [0,2707]
    matrix[x][y] = matrix[y][x] = 1
print(sum(matrix))

# reshaping the array from 3D
# matrice to 2D matrice.
edges_reshaped = edges.reshape(edges.shape[0], -1)
# saving reshaped array to file.
# np.savetxt("data/cora/cora_edges_dim1.txt", edges)
np.savetxt("data/cora/cora_edges49.txt", edges_reshaped)

# retrieving data from file.
loaded_arr = np.loadtxt("data/cora/cora_edges49.txt")

# This loadedArr is a 2D array, therefore
# we need to convert it to the original
# array shape.reshaping to get original
# matrice with original shape.
load_original_arr = loaded_arr.reshape(
	loaded_arr.shape[0], loaded_arr.shape[1] // edges.shape[2], edges.shape[2])

# # check the shapes:
# print("shape of arr: ", edges.shape)
# print("shape of load_original_arr: ", load_original_arr.shape)
#
# # check if both arrays are same or not:
# if (load_original_arr == edges).all():
# 	print("Yes, both the arrays are same")
# else:
# 	print("No, both the arrays are not same")


np.savetxt("data/cora/cora_adj.txt", matrix)
np.savetxt("data/cora/cora_labels.txt", list_labels)
np.savetxt("data/cora/cora_features.txt", features)