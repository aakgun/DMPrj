



import networkx as nx
from scipy.sparse.linalg import svds
import numpy as np
from bionev.OpenNE import gf, grarep,graph, hope, lap, line, node2vec, sdne
from bionev.OpenNE import hope,gf,grarep,lap,sdne,graph

###################

import networkx as nx

import numpy as np





size=100

node_list = list(G.nodes())
adjacency_matrix = nx.adjacency_matrix(G, node_list)
adjacency_matrix = adjacency_matrix.astype(float)

U, Sigma, VT = svds(adjacency_matrix, size)
Sigma = np.diag(Sigma)
W = np.matmul(U, np.sqrt(Sigma))
C = np.matmul(VT.T, np.sqrt(Sigma))
# print(np.sum(U))
embeddings = W + C
vectors = {}

np.array(embeddings[0])

for id, node in enumerate(node_list):
        vectors[node] = list(np.array(embeddings[id]))


output_filename = "SVD_model_output"
fout = open(output_filename, 'w')
node_num = len(vectors.keys())
fout.write("{} {}\n".format(node_num, size))
for node, vec in vectors.items():
    fout.write("{} {}\n".format(node,
                    ' '.join([str(x) for x in vec])))
    print("{} {}\n".format(node,
                    ' '.join([str(x) for x in vec])))

print(type(G))



A.save_embeddings ("SaveHopeEmbedding.txt")


class CustomGraph(object):
    desc ="Custom Grpah"

    def __init__(self,graph):
        self.nodesize1=100
        self.nodesizeGraph=len(graph.nodes)+2000

Gnx1=nx.read_edgelist(train_graph_filename)
B=CustomGraph(Gnx1)
print(B.desc)
print(B.nodesizeGraph)

        
time1 = time.time()
_results = dict(
        )



train_graph_filename= "C:/git/BioNEV/data/DrugBank_DDI/DrugBank_DDI.edgelist"
train_graph_filename= "C:/git/BioNEV/data/DrugBank_DDI/Karate.edgelist"
G2=Graph()
#G..read_weighted_edgelist(train_graph_filename)
G1 = nx.read_weighted_edgelist(train_graph_filename)
G2.read_g(G1)

from bionev.OpenNE import node2vec
from bionev.OpenNE import gf, grarep, hope, lap, line, node2vec, sdne
from bionev.utils import *

G, G_train, testing_pos_edges, train_graph_filename = split_train_test_graph(args.input, args.seed, weighted=args.weighted)
G, G_train, testing_pos_edges, train_graph_filename = split_train_test_graph(train_graph_filename, None, weighted=False)



model = node2vec.Node2vec(graph=G_, path_length=args.walk_length,
                                      num_paths=args.number_walks, dim=args.dimensions,
                                      workers=args.workers, p=args.p, q=args.q, window=args.window_size)


g1 = read_for_OpenNE(train_graph_filename, weighted=False)
model = node2vec.Node2vec(g1,path_length=64,num_paths=32,dim=100, workers=8,p=1,q=1,window=10)
print("test")

model = node2vecmodel.fit(window=10,min_count=1)
# Save embeddings for later use
model.wv.save_word2vec_format("NODE2VEC_EMBEDDING_FILENAME")

model.wv.most_similar('6') 

for node, _ in model.most_similar('6'):
    print(node)

from tsne import tsne
from sklearn.manifold import TSNE
import numpy as np
import pylab
import pandas as pd
import matplotlib as plt
from sklearn import decomposition
pca = decomposition.PCA()
# configuring the parameteres
# the number of components = 2
pca.n_components = 2
pca_data = pca.fit_transform(X)
print(X)
print("shape of pca_reduced.shape = ", pca_data.shape)


from sklearn.manifold import TSNE
tsne = TSNE(random_state=1, n_iter=15000, metric="cosine")
X = np.loadtxt("NODE2VEC_EMBEDDING_FILENAME", skiprows=1)
X = np.array([x[1:] for x in X])
Label = np.loadtxt("NODE2VEC_EMBEDDING_FILENAME", skiprows=1)
Label = np.array([int(x[0]) for x in Label])

embs = tsne.fit_transform(X)
df=pd.DataFrame()
df['x'] = embs[:, 0]
df['y'] = embs[:, 1]
df['label'] = list(Label)
print(df.shape)
print(df['y'])
print(df['x'])
print(X[0])

FS = (10, 8)
fig, ax = plt.subplots(figsize=FS)
# Make points translucent so we can visually identify regions with a high density of overlapping points
ax.scatter(df.x, df.y,label=df.label, alpha=.1);
#ax.legend()
#ax.annotate(df.x, df.y, Label, fontsize=9)
for i, txt in enumerate(Label):
    ax.annotate(txt, (df.x[i], df.y[i]))

dfselection=df[(df.x > -30) & (df.x <-10 ) & (df.y <-45 )]    
print(dfselection)

X = np.loadtxt("NODE2VEC_EMBEDDING_FILENAME", skiprows=1)
X = np.array([x[1:] for x in X])
Y = TSNE(X, 2, 50, 20.0)
pylab.scatter(Y[:, 0], Y[:, 1], 20)
pylab.show()

import bionev.utils


from bionev.evaluation import LinkPrediction, NodeClassification
from bionev.embed_train import embedding_training, load_embedding, read_node_labels, split_train_test_graph
from bionev.SVD.model import SVD_embedding
from bionev.utils import *

from utils2
G, G_train, testing_pos_edges, train_graph_filename = split_train_test_graph(train_graph_filename, None, weighted=None)


GSnap, GNx= read_GRaph_SNAP_NX()

g = read_for_SVD(train_graph_filename, weighted=None)

SVD_embedding(g, "SVD_OutputFile", size=100)
embedding_look_up = load_embedding("SVD_OutputFile")
result = LinkPrediction(embedding_look_up, G, G_train, testing_pos_edges,0)
auc_roc, auc_pr, accuracy, f1 = result
_results['results'] = dict(
                auc_roc=auc_roc,
                auc_pr=auc_pr,
                accuracy=accuracy,
                f1=f1,
            )
print(_results)

import time
G, G_train, testing_pos_edges, train_graph_filename = split_train_test_graph(train_graph_filename, None, weighted=None)
g = read_for_OpenNE(train_graph_filename, weighted=None)
model = hope.HOPE(g, 100)
model.save_embeddings("HOPE_OutputFile")
embedding_look_up = load_embedding("HOPE_OutputFile")
result = LinkPrediction(embedding_look_up, G, G_train, testing_pos_edges,0)
auc_roc, auc_pr, accuracy, f1 = result
_results['results'] = dict(
                auc_roc=auc_roc,
                auc_pr=auc_pr,
                accuracy=accuracy,
                f1=f1,
            )
print(_results)

G, G_train, testing_pos_edges, train_graph_filename = split_train_test_graph(train_graph_filename, None, weighted=None)
g = read_for_OpenNE(train_graph_filename, weighted=None)
model = lap.LaplacianEigenmaps(g, rep_size=100)
model.save_embeddings("Laplacian_OutputFile")
embedding_look_up = load_embedding("Laplacian_OutputFile")
result = LinkPrediction(embedding_look_up, G, G_train, testing_pos_edges,0)
auc_roc, auc_pr, accuracy, f1 = result
_results['results'] = dict(
                auc_roc=auc_roc,
                auc_pr=auc_pr,
                accuracy=accuracy,
                f1=f1,
            )
print(_results)




        embed_train_time = time.time() - time1
        print('Embedding Learning Time: %.2f s' % embed_train_time)
        embedding_look_up = load_embedding(args.output)
        time1 = time.time()
        print('Begin evaluation...')
        result = LinkPrediction(embedding_look_up, G, G_train, testing_pos_edges,args.seed)
        eval_time = time.time() - time1
        print('Prediction Task Time: %.2f s' % eval_time)






















model = HOPE(G, d=100)
A = nx.to_numpy_matrix(g)

# self._beta = 0.0728

# M_g = np.eye(graph.number_of_nodes()) - self._beta * A
# M_l = self._beta * A

M_g = np.eye(graph.number_of_nodes())
M_l = np.dot(A, A)

S = np.dot(np.linalg.inv(M_g), M_l)
# s: \sigma_k
u, s, vt = lg.svds(S, k=self._d // 2)
sigma = np.diagflat(np.sqrt(s))
X1 = np.dot(u, sigma)
X2 = np.dot(vt.T, sigma)


from SVD.model import SVD_embedding
SVD_embedding(G, "SVD_model_output", 100)


model = node2vec.Node2vec(graph=G, path_length=args.walk_length,
                                      num_paths=args.number_walks, dim=args.dimensions,
                                      workers=args.workers, window=args.window_size, dw=True)


