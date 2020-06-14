from keras.layers import Dense, LSTM, Embedding
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from sklearn.model_selection import train_test_split
import gzip
import pandas as pd
import os

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
        print('\n' + str(cat1) + '({}) --> '.format(str(cat_counts[cat1])), end='')
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

def build_model2(embedding_matrix,categories):
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], embedding_size, weights=[embedding_matrix],input_length=maxlen,trainable=False))
    model.add(LSTM(32))
#   We don't lose much by replacing LSTM with this flatten layer (as we have short sequences)
#   model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(categories), activation='sigmoid'))
    model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
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

def read_data2():
    df_train = train
    df_val = test
    #categories = list(set(df_train.categories.values))
    categories = list(set(dfdrugbank2.organism_code.values))
    #categories = list(set(df.cat2id.values))
    return df_train,df_val,categories

def read_data3():
    df_train = train
    df_val = test
    #categories = list(set(df_train.categories.values))
    categories = list(set(dfdrugbank2.organism.values))
    categories = list(set(dfdrugbank2.organism_code.values))
    #categories = list(set(df.cat2id.values))
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
    y_train = category_to_one_hot(df_train['organism_code'].values,categories)
    y_val = category_to_one_hot(df_val['organism'].values,categories)
    return word_index,x_train,y_train,x_val,y_val

    #df_train,df_val,categories = read_data2()
    ##Load Glove 50-d embeddings
    #embeddings,word_index = load_embeddings()
    #tk_word_index,x_train,y_train,x_val,y_val = prepare_data_from_full_word_index2(df_train,df_val,categories,word_index)

def prepare_data_from_full_word_index2(df_train,df_val,categories,word_index):
    train_text = df_train['description'].tolist()
    val_text = df_val['description'].tolist()
    x_train = get_pad_sequences(train_text,word_index)
    x_val = get_pad_sequences(val_text,word_index)
    y_train = category_to_one_hot(df_train['organism_code'].values,categories)
    #y_train = df_train['cat2id']
    #y_val = df_val['cat2id']
    y_val = category_to_one_hot(df_val['organism_code'].values,categories)
    return word_index,x_train,y_train,x_val,y_val

def prepare_data_from_full_word_index3(df_train,df_val,categories,word_index):
    train_text = df_train['description'].tolist()
    val_text = df_val['description'].tolist()
    x_train = get_pad_sequences(train_text,word_index)
    x_val = get_pad_sequences(val_text,word_index)
    #y_train = category_to_one_hot(df_train['organism'].values,categories)
    y_train = df_train['organism_code']
    #y_train = df_train['cat2id']
    #y_val = df_val['cat2id']
    #y_val = category_to_one_hot(df_val['organism'].values,categories)
    y_val = df_val['organism_code']
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

def category_to_id(cat_desc):
    catidx=None
    for id,val in enumerate(cat_desc):
#        print(id,val)
        catidx=catlist2.index(val)
#        print(catidx)
    return (catidx)


Homedir = os.getcwd()
dfdrugbank_filename=os.path.join(Homedir, './data/drugbank.tsv')
dfprotein_filename=os.path.join(Homedir, './data/proteins.tsv')
dfnodes_filename=os.path.join(Homedir, './data/DrugBank_DDI/node_list.txt')
dfdrugbank = pd.read_csv(dfdrugbank_filename, delimiter="\t")
dfprotein = pd.read_csv(dfprotein_filename, delimiter="\t")
dfdrugbank['DrugBank_id'] = dfdrugbank['drugbank_id']
dfprotein['DrugBank_id'] = dfprotein['drugbank_id']
dfnodes = pd.read_csv(dfnodes_filename, delimiter="	")
dfnodes['ID'] = dfnodes.index
dfnodes['NODEID'] = dfnodes['ID']
dfnodes['WEIGHT'] = 0
dfdrugbank2 = dfdrugbank.merge(dfnodes, how="left", on="DrugBank_id")
dfdrugbank2 = dfdrugbank2.merge(dfprotein, how="left", on="DrugBank_id")
#dfdrugbank3 = dfdrugbank2[str(dfdrugbank2['ID'])!="nan"]
dfdrugbank2.dropna(inplace=True)
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
dfdrugbank2["category_code"] = lb_make.fit_transform(dfdrugbank2["category"])
dfdrugbank2["actions_code"] = lb_make.fit_transform(dfdrugbank2["actions"])
dfdrugbank2["organism_code"] = lb_make.fit_transform(dfdrugbank2["organism"])
dfdrugbank2["groups_code"] = lb_make.fit_transform(dfdrugbank2["groups"])
dfdrugbank2['ID'] = dfdrugbank2['ID'].astype(int)
print(dfdrugbank2.head(5))
print(dfdrugbank2.columns)
print("test")


# dfdrugbank=pd.read_csv("C:/git/drugbank/data/drugbank.tsv",delimiter="\t")
# dfprotein=pd.read_csv("C:/git/drugbank/data/proteins.tsv",delimiter="\t")
# dfdrugbank['DrugBank_id']=dfdrugbank['drugbank_id']
# dfprotein['DrugBank_id']=dfprotein['drugbank_id']
# dfnodes=pd.read_csv("C:/git/BioNEV/data/DrugBank_DDI/node_list.txt",delimiter="	")
# dfnodes['ID']=dfnodes.index
# from sklearn.preprocessing import LabelEncoder
# lb_make = LabelEncoder()
# dfdrugbank2["category_code"] = lb_make.fit_transform(dfdrugbank2["category"])
# dfdrugbank2["actions_code"] = lb_make.fit_transform(dfdrugbank2["actions"])
# dfdrugbank2["organism_code"] = lb_make.fit_transform(dfdrugbank2["organism"])
# dfdrugbank2["groups_code"] = lb_make.fit_transform(dfdrugbank2["groups"])
# dfdrugbank2=dfdrugbank.merge(dfnodes,how="left",on="DrugBank_id")
# dfdrugbank2=dfdrugbank.merge(dfprotein,how="left",on="DrugBank_id")

protein_category=list(dfdrugbank2.category)
protein_category = set(protein_category)
protein_category =list(protein_category)

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

df = dfdrugbank2[pd.notnull(dfdrugbank2['description'])]
df = df[pd.notnull(df['description'])]

train, test  = train_test_split( df, test_size=0.20, random_state=10)

vocab_size = 400000
embedding_size = 50
maxlen = 10
embeddings_path = 'C:/git/BioNEV/embeddings/glove.6B.50d.txt.gz'
import numpy as np

print(categories)

df_train,df_val,categories = read_data2()
df_train,df_val,categories = read_data3()

#Load Glove 50-d embeddings
embeddings,word_index = load_embeddings()
tk_word_index,x_train,y_train,x_val,y_val = prepare_data_from_full_word_index2(df_train,df_val,categories,word_index)
#tk_word_index,x_train,y_train,x_val,y_val = prepare_data_from_full_word_index3(df_train,df_val,categories,word_index)

# Get the embedding matrix for the model, build model, display model summary
embedding_matrix = get_embedding_matrix_for_model(embeddings,word_index)
model = build_model(embedding_matrix,categories)

# Train the model, record history
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=24,
                    shuffle=False,
                    validation_data=(x_val, y_val))




import matplotlib.pyplot as plt
make_history_plot(history)

# Make and analyze training predictions
train_predictions = one_hot_to_category(model.predict(x_train),categories)
train_predictions = model.predict(x_train)
analyze_predictions(categories,df_train['organism_code'].values,train_predictions)

# Make and analyze validation predictions
val_predictions = one_hot_to_category(model.predict(x_val),categories)
analyze_predictions(categories,df_val['organism'].values,val_predictions)
