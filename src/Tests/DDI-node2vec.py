#model = node2vec.Node2vec(graph=G_, path_length=args.walk_length,
#                                      num_paths=args.number_walks, dim=args.dimensions,
#                                      workers=args.workers, p=args.p, q=args.q, window=args.window_size)

from bionev import *
from bionev.OpenNE import node2vec
from bionev.OpenNE import gf, grarep, hope, lap, line, node2vec, sdne

import networkx as nx
from nltk.corpus import stopwords
from scipy.sparse.linalg import svds
import numpy as np

from bionev.OpenNE import gf, grarep,graph, hope, lap, line, node2vec, sdne
from bionev.utils import *
from bionev.evaluation import LinkPrediction, NodeClassification
from bionev.embed_train import embedding_training, load_embedding, read_node_labels, split_train_test_graph
from bionev.SVD.model import SVD_embedding
from bionev.utils import *

train_graph_filename= "C:/git/BioNEV/data/DrugBank_DDI/Karate.edgelist"
train_graph_filename= "C:/git/BioNEV/data/DrugBank_DDI/DrugBank_DDI.edgelist"

import matplotlib.pyplot as plt
from bionev.utils import *

#Gnx1=nx.read_edgelist(train_graph_filename)
#nx.draw(Gnx1, pos=nx.spring_layout(Gnx1),with_labels = True)  # use spring layout

g1 = read_for_OpenNE(train_graph_filename, weighted=False)


from bionev.OpenNE import gf, grarep,graph, hope, lap, line, node2vec, sdne
from bionev.OpenNE import node2vec
#model = node2vec.Node2vec(g1,path_length=64,num_paths=32,dim=100,workers=8,p=1,q=1,window=10)
#model = node2vec.Node2vec(g1,path_length=32,num_paths=16,dim=20,workers=8,p=1,q=1,window=10)


model = node2vec.Node2vec(g1,path_length=10,num_paths=5,dim=100,workers=8,p=1,q=2,window=10)
model.save_embeddings("C:/git/BioNEV/embeddings/NODE2VEC_OutputFile")
print('tests')


EMBED="NODE2VEC_EMBEDDING_FILE"
EMBED="C:/git/BioNEV/embeddings/Laplacian_OutputFile"
EMBED="C:/git/BioNEV/embeddings/HOPE_OutputFile"
EMBED="C:/git/BioNEV/embeddings/NODE2VEC_OutputFile"
from sklearn.manifold import TSNE
tsne = TSNE(random_state=1, n_iter=15000, metric="cosine")
X = np.loadtxt(EMBED, skiprows=1)
X = np.array([x[1:] for x in X])
Label = np.loadtxt(EMBED, skiprows=1)
Label = np.array([int(x[0]) for x in Label])

import pandas as pd
embs = tsne.fit_transform(X)
df=pd.DataFrame()
df['x'] = embs[:, 0]
df['y'] = embs[:, 1]
df['label'] = list(Label)
print(df.shape)
print(df['y'])
print(df['x'])
print(X[0])

import matplotlib.pyplot as plt

FS = (10, 8)
fig, ax = plt.subplots(figsize=FS)
# Make points translucent so we can visually identify regions with a high density of overlapping points
ax.scatter(df.x, df.y,label=df.label, alpha=.1);
#ax.legend()
#ax.annotate(df.x, df.y, Label, fontsize=9)
for i, txt in enumerate(Label):
    ax.annotate(txt, (df.x[i], df.y[i]))

plt.savefig(EMBED + ".png")
plt.show()

for i, txt in enumerate(Label):
    if (df.x[i] > -20 and df.x[i] <20 and df.y[i]>-75 and df.y[i]<-25):
        #print(txt, (df.x[i], df.y[i]))
        if drug2_to_index.get(index_to_drug[txt],None)!=None:
                    print(txt,index_to_drug[txt],
                          dfdrugbank2.iloc[drug2_to_index[index_to_drug[txt]]].categories,
                          (df.x[i], df.y[i]))
        else:
            print(txt, "NoneDrug",(df.x[i], df.y[i]))

drug2_to_index[index_to_drug[1750]]
d.get("A",None)


def default_clean(text):
    '''
    Removes default bad characters
    '''
    if not (pd.isnull(text)):
            # text = filter(lambda x: x in string.printable, text)
        bad_chars = set(
                ["@", "+", '/', "'", '"', '\\', '(', ')', '', '\\n', '', '?', '#', ',', '.', '[', ']', '%', '$', '&', ';', '!',
         ';', ':', "*", "_", "=", "}", "{"])
        for char in bad_chars:
            text = text.replace(char, " ")
            text = re.sub('\d+', "", text)
    return text

from nltk.stem.porter import *
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
##################################################
def get_pad_sequences(text_list,word_index):
    seqs = []
    for text in text_list:
        word_seq = text_to_word_sequence(text.lower())
        seq = []
        for word in word_seq:
          if word in word_index:
            seq.append(word_index[word])
        seqs.append(seq)
    return pad_sequences(seqs,maxlen)
#################################################
train_text = dfdrugbank2['description'].tolist()

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import gutenberg
import gensim
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
data = []
for i in sent_tokenize(train_text):
    temp = []
    for j in tokenizer.tokenize(i):
        temp.append(j.lower())
    data.append(temp)

x_train = get_pad_sequences(train_text,word_index)
from keras.preprocessing.text import Tokenizer,text_to_word_sequence

###########################
textdata = []
for doc in dfdrugbank2.index:
  textdata.append(dfdrugbank2.iloc[doc].description)
tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))

import math
def nlp_clean(x):
    #x=default_clean(x)
    new_data = []
    for d in x:
        if str(d)!="nan":
            new_str = d.lower()
            dlist = tokenizer.tokenize(new_str)
            dlist = list(set(dlist).difference(stopword_set))
            new_data.append(dlist)
            #new_data.append(d)
            print(d,i)

        #d = re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", d)

    return new_data

textdata1 = nlp_clean(textdata)


#iterator returned over all documents
docLabels = list(dfdrugbank2['type'])
it = LabeledLineSentence(textdata1, docLabels)

modelgenDoc2Vec = gensim.models.Doc2Vec(vector_size=300, min_count=0, alpha=0.025, min_alpha=0.025)
modelgenDoc2Vec = gensim.models.Doc2Vec(vector_size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025,total_examples=modelgenDoc2Vec.corpus_count) # use fixed learning rate
modelgenDoc2Vec.build_vocab(it)

modelgenDoc2Vec = Doc2Vec(size=100, window=10, min_count=5, workers=11,alpha=0.025, iter=20)
modelgenDoc2Vec.build_vocab(sentences)
modelgenDoc2Vec.train(it,total_examples=model.corpus_count, epochs=modelgenDoc2Vec.iter)


model = Doc2Vec(size=100, window=10, min_count=5, workers=11,alpha=0.025, iter=20)
model.build_vocab(it)
model.train(it,total_examples=model.corpus_count, epochs=model.iter)
model.save('C:/git/BioNEV/embeddings/model_docsimilarity.doc2vec')

#training of model
for epoch in range(100):
        print ('iteration '+str(epoch+1))
        modelgenDoc2Vec.train(it)
        momodelgenDoc2Vecdel.alpha -= 0.002
        modelgenDoc2Vec.min_alpha = modelgenDoc2Vec.alpha
#saving the created model
model.save('doc2vec.model')
print “model saved”

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,[self.labels_list[idx]])
###############################################################

dfdrugbank2['description'] = dfdrugbank2['description'].apply(lambda x:default_clean(x))
dfdrugbank2['description'] = dfdrugbank2['description'].apply(lambda x:stop_and_stem(x))

dfdrugbank2['test'] = dfdrugbank2['description'].apply(lambda x:default_clean(x))
dfdrugbank2['test'] = dfdrugbank2['description'].apply(lambda x:stop_and_stem(x))

dfdrugbank2['test2'] = dfdrugbank2['description'].apply(lambda x:stop_and_stem(x))
dfdrugbank2['test2'] = dfdrugbank2['test2'].apply(lambda x:default_clean(x))
dfdrugbank2['test2'] = dfdrugbank2['test2'].apply(lambda x:build_corpus(x))

def build_corpus(text):
    words = []
    if str(text)!="nan":
        porter = nltk.PorterStemmer()

        tokenizer = RegexpTokenizer(r'\w+')
        for i in sent_tokenize(text):
            for j in tokenizer.tokenize(i):
                words.append(j.lower())
        words = np.unique(words)
    return words

text = nltk.corpus.gutenberg.raw('austen-emma.txt')
text1 = stop_and_stem(text)

def stop_and_stem(text, stem=True, stemmer=PorterStemmer()):
    '''
    Removes stopwords and does stemming
    '''
    stoplist = stopwords.words('english')
        if stem:
            if str(text)!="nan":
                #text = re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", text)
                text=re.sub(r"\d", "", text)
                text_stemmed = [stemmer.stem(word) for word in word_tokenize(text) if word not in stoplist and len(word) >3]
        else:
            text_stemmed = [word for word in word_tokenize(text) if word not in stoplist and len(word) >3]


        if str(text) != "nan":
            text = ' '.join(text_stemmed)
        return text

class TaggedDocumentIterator(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=doc.split(), tags=[self.labels_list[idx]])

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

docLabels = list(dfdrugbank2['type'])
data = list(dfdrugbank2['test'])
data=nlp_clean(data)
sentences = TaggedDocumentIterator(data, docLabels)

data2 = list(dfdrugbank2['test2'])
docLabels2 = list(dfdrugbank2['type'])
sentences2 = TaggedDocumentIterator(data2, docLabels2)


modelD2V = Doc2Vec(vector_size=100, window=10, min_count=5, workers=11,alpha=0.025, epochs=20)
modelD2V.build_vocab(sentences2)
modelD2V.train(sentences2,total_examples=modelD2V.corpus_count, epochs=modelD2V.iter)

modelD2V = Doc2Vec(vector_size=100, window=10, min_count=5, workers=11,alpha=0.025, epochs=20)
modelD2V.build_vocab(xlist)
modelD2V.train(xlist,total_examples=modelD2V.corpus_count, epochs=modelD2V.iter)


documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(dfdrugbank2.description)]
modelD2V = Doc2Vec(dm=0, # DBOW
				size=400,
				window=8,
				min_count=5,
				dbow_words = 1) # DBOW, simultaneously train word vectors with doc vectors

modelD2V = Doc2Vec(epochs=100, window=10, min_count=5, workers=11,alpha=0.025, iter=20)
modelD2V.save("C:/git/BioNEV/embeddings/Doc2Vec_OutputFile")
distance = modelD2V.wv.n_similarity(documents[0],documents[1])

print(drug2_to_index.get(index_to_drug[1750],None))
print(drug2_to_index.get(index_to_drug[1750],None))

stoplist = stopwords.words('english')
stemmer=nltk.PorterStemmer
wordlist=[]
for row in dfdrugbank2.index:
    indexwordlist=[]
    text=dfdrugbank2[row].desription
    if str(text)!="nan":
        #text = re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", text)
        text=re.sub(r"\d", "", text)
        text_stemmed = [stemmer.stem(word) for word in word_tokenize(text) if word not in stoplist and len(word) >3]
        indexwordlist.append(text_stemmed)
    wordlist.append(j(text_stemmed))

[stemmer.stem(word) for word in word_tokenize(text) if word not in stoplist and len(word) >3]

xlist=[]
xlist.append(list(dfdrugbank2['description'].apply(lambda x:x.split(" ") if str(x) != "nan" else "" )))
#xlist.append(list(dfdrugbank2['description'].apply(lambda x:[stemmer.stem(word) for word in word_tokenize(x) if word not in stoplist and len(word) >3] if str(x) != "nan" else "" )))
#xlist.append(list([stemmer.stem(word) for word in word_tokenize(dfdrugbank2['groups']) if word not in stoplist and len(word) >3 and str(dfdrugbank2['groups'])!="nan"]))
xlist.append(list(dfdrugbank2['groups']))

##################################
from keras.layers import Dense, LSTM, Embedding
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from sklearn.model_selection import train_test_split
import gzip

df = dfdrugbank2[pd.notnull(dfdrugbank2['description'])]
df = df[pd.notnull(df['description'])]

train, test  = train_test_split( df, test_size=0.20, random_state=10)

vocab_size = 400000
embedding_size = 50
maxlen = 10
embeddings_path = 'C:/git/BioNEV/embeddings/glove.6B.50d.txt.gz'

df_train,df_val,categories = read_data()
#Load Glove 50-d embeddings
embeddings,word_index = load_embeddings()
tk_word_index,x_train,y_train,x_val,y_val = prepare_data_from_full_word_index(df_train,df_val,categories,word_index)

# Get the embedding matrix for the model, build model, display model summary
embedding_matrix = get_embedding_matrix_for_model(embeddings,word_index)
model = build_model(embedding_matrix,categories)


# Train the model, record history
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=24,
                    shuffle=False,
                    validation_data=(x_val, y_val))



make_history_plot(history)

# Make and analyze training predictions
train_predictions = one_hot_to_category(model.predict(x_train),categories)
analyze_predictions(categories,df_train['categories'].values,train_predictions)

# Make and analyze validation predictions
val_predictions = one_hot_to_category(model.predict(x_val),categories)
analyze_predictions(categories,df_val['categories'].values,val_predictions)

def get_val(numerator,divisor):
    return float('nan') if divisor == 0 else np.round(numerator/divisor,3)
from collections import defaultdict
def analyze_predictions(categories, y_true, y_pred):
    tp = defaultdict(int)
    tn = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    precisions = []
    recalls = []
    f1s = []
    cat_counts = defaultdict(int)
    for cat in y_true:
        cat_counts[cat] += 1
    correct = 0
    conf_mat = defaultdict(dict)
    for cat1 in categories:
        for cat2 in categories:
            conf_mat[cat1][cat2] = 0
    for y, y_hat in zip(y_true, y_pred):
        conf_mat[y][y_hat] += 1
        if y == y_hat:
            correct += 1
            tp[y] += 1
        else:
            fp[y_hat] += 1
            fn[y] += 1
    print('Overall Accuracy:', round(correct / len(y_pred), 3))
    for cat in categories:
        precision = get_val(tp[cat], tp[cat] + fp[cat])
        recall = get_val(tp[cat], (tp[cat] + fn[cat]))
        f1 = get_val(2 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        print('{} --> Precision:{},Recall:{},F1:{}'.format(cat, precision, recall, f1))
    print('\nAverages---> Precision:{}, Recall:{}, F1:{}'.format(np.round(np.nanmean(precisions), 3),
                                                                 np.round(np.nanmean(recalls), 3),
                                                                 np.round(np.nanmean(f1s), 3))
          )

    print('\nConfusion Matrix')
    for cat1 in categories:
        print('\n' + cat1 + '({}) --> '.format(cat_counts[cat1]), end='')
        for cat2 in categories:
            print('{}({})'.format(cat2, conf_mat[cat1][cat2]), end=' , ')
    print('')


def one_hot_to_category(cat_one_hot_list,cat_master):
    return [cat_master[cat_one_hot.argmax()] for cat_one_hot in cat_one_hot_list]

# From Deep Learning with Python book
def make_history_plot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', color='green',label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', color='green',label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
# Build the keras model
def build_model(embedding_matrix,categories):
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], embedding_size, weights=[embedding_matrix],input_length=maxlen,trainable=False))
    model.add(LSTM(32))
#   We don't lose much by replacing LSTM with this flatten layer (as we have short sequences)
#   model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(categories), activation='sigmoid'))
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
    return model

def get_embedding(word,word_index,embeddings):
    if word in word_index:
        return embeddings[word_index[word]].reshape(((embedding_size,1)))
    else:
        return np.zeros((embedding_size,1))
# Get the embedding weights for the model
def get_embedding_matrix_for_model(embeddings,word_index):
    train_val_words = min(vocab_size, len(word_index)) +1
    embedding_matrix = np.zeros((train_val_words, embedding_size))
    for word, i in word_index.items():
        embedding_vector = get_embedding(word,word_index,embeddings).flatten()
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
# Common methods to read data and get embeddings
def read_data():
    df_train = train
    df_val = test
    categories = list(set(df_train.categories.values))
    return df_train,df_val,categories

def load_embeddings():
    word_index = {}
    embeddings = np.zeros((vocab_size,embedding_size))
    with gzip.open(embeddings_path) as file:
        for i,line in enumerate(file):
            line_tokens = line.split()
            word = line_tokens[0].decode('utf8')
            embeddings[i] = np.asarray(line_tokens[1:],dtype='float32')
            word_index[word] = i
    return embeddings,word_index

def prepare_data_from_full_word_index(df_train,df_val,categories,word_index):
    train_text = df_train['description'].tolist()
    val_text = df_val['description'].tolist()
    x_train = get_pad_sequences(train_text,word_index)
    x_val = get_pad_sequences(val_text,word_index)
    y_train = category_to_one_hot(df_train['categories'].values,categories)
    y_val = category_to_one_hot(df_val['categories'].values,categories)
    return word_index,x_train,y_train,x_val,y_val

# Convert the list of categories to one_hot vector
def category_to_one_hot(cat_list,cat_master):
    cat_dict = {}
    for i,cat in enumerate(cat_master):
        cat_dict[cat] = i
    cat_integers = [cat_dict[cat] for cat in cat_list]
    return to_categorical(cat_integers,num_classes=len(cat_master))


def get_pad_sequences(text_list,word_index):
    seqs = []
    for text in text_list:
        word_seq = text_to_word_sequence(text.lower())
        seq = []
        for word in word_seq:
          if word in word_index:
            seq.append(word_index[word])
        seqs.append(seq)
    return pad_sequences(seqs,maxlen)

##################################



dfdrugbank2.iloc[drug2_to_index.get(index_to_drug[txt],None)].categories

dfdrugbank=pd.read_csv("C:/git/drugbank/data/drugbank.tsv",delimiter="\t")
dfdrugbank['DrugBank_id']=dfdrugbank['drugbank_id']
dfnodes=pd.read_csv("C:/git/BioNEV/data/DrugBank_DDI/node_list.txt",delimiter="	")
dfnodes['ID']=dfnodes.index
#print(dfnodes.iloc[0].DrugBank_id)

drug_to_index = {}
index_to_drug = {}
counter = 0

for node in dfnodes.index:
    #print(dfnodes.iloc[node].DrugBank_id)
    #for w in words:
     drug_to_index[dfnodes.iloc[node].DrugBank_id] = node
     index_to_drug[node] = dfnodes.iloc[node].DrugBank_id


drug2_to_index = {}
index_to_drug2 = {}

for node in dfdrugbank2.index:
    #print(dfdrugbank2.iloc[node].DrugBank_id,dfdrugbank2.iloc[node].type)
    drug2_to_index[dfdrugbank2.iloc[node].DrugBank_id] = node
    index_to_drug2[node] = dfdrugbank2.iloc[node].DrugBank_id

print(drug_to_index['DB09020'])
print(drug2_to_index['DB09020'])

print(drug_to_index['DB09079'])
print(drug2_to_index['DB09079'])



print(index_to_drug[0])
print(drug2_to_index[index_to_drug[0]])
print(dfdrugbank2.iloc[drug2_to_index[index_to_drug[0]]].categories)

dfdrugbank2=dfdrugbank.merge(dfnodes,how="left",on="DrugBank_id")

dfdrugbank2.iloc[45].DrugBank_id



############################################################################################

from gensim.models import Word2Vec
from bionev.OpenNE import walker

modelwalker = walker.Walker(g1, p=1, q=0.5, workers=8)
print("Preprocess transition probs...")
modelwalker.preprocess_transition_probs()
#sentences = modelwalker.simulate_walks(num_walks=32, walk_length=64)
#sentences = modelwalker.simulate_walks(num_walks=1, walk_length=10)

sentences1,sentences2 = modelwalker.simulate_walks2(num_walks=1, walk_length=10)

kwargs={}
kwargs["sentences"] = sentences
kwargs["min_count"] = 0
kwargs["size"] = 100
kwargs["sg"] = 1

print("Learning representation...")
word2vec = Word2Vec(**kwargs)
vectors = {}
for word in g1.G.nodes():
    vectors[word] = word2vec.wv[word]
del word2vec

filename="C:/git/BioNEV/src/node2vec Embeddings.txt"
fout = open(filename, 'w')
node_num = len(vectors.keys())
fout.write("{} {}\n".format(node_num, 100))
for node, vec in vectors.items():
    fout.write("{} {}\n".format(node,
                                ' '.join([str(x) for x in vec])))
fout.close()

############################################################################################
_results = dict()

G, G_train, testing_pos_edges, train_graph_filename = split_train_test_graph(train_graph_filename, None, weighted=None)
g = read_for_SVD(train_graph_filename, weighted=None)

EMBED="C:/git/BioNEV/embeddings/SVD_OutputFile"

SVD_embedding(g, EMBED, size=100)
embedding_look_up = load_embedding(EMBED)
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
EMBED="C:/git/BioNEV/embeddings/HOPE_OutputFile"

g = read_for_OpenNE(train_graph_filename, weighted=None)
model = hope.HOPE(g, 100)
model.save_embeddings("C:/git/BioNEV/embeddings/HOPE_OutputFile")
embedding_look_up = load_embedding("C:/git/BioNEV/embeddings/HOPE_OutputFile")
result = LinkPrediction(embedding_look_up, G, G_train, testing_pos_edges,0)
auc_roc, auc_pr, accuracy, f1 = result
_results['results'] = dict(
                auc_roc=auc_roc,
                auc_pr=auc_pr,
                accuracy=accuracy,
                f1=f1,
            )
print(_results)

EMBED="C:/git/BioNEV/embeddings/Laplacian_OutputFile"
G, G_train, testing_pos_edges, train_graph_filename = split_train_test_graph(train_graph_filename, None, weighted=None)
g = read_for_OpenNE(train_graph_filename, weighted=None)
model = lap.LaplacianEigenmaps(g, rep_size=100)
model.save_embeddings(EMBED)
embedding_look_up = load_embedding(EMBED)
result = LinkPrediction(embedding_look_up, G, G_train, testing_pos_edges,0)
auc_roc, auc_pr, accuracy, f1 = result
_results['results'] = dict(
                auc_roc=auc_roc,
                auc_pr=auc_pr,
                accuracy=accuracy,
                f1=f1,
            )
print(_results)