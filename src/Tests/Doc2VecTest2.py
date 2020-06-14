from keras.layers import Dense, LSTM, Embedding
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from sklearn.model_selection import train_test_split
import gzip
import pandas as pd
import os

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

df=dfdrugbank2

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt

cnt_pro = df['category'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Product', fontsize=12)
plt.xticks(rotation=90)
plt.show();


from bs4 import BeautifulSoup
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text
df['description'] = df['description'].apply(cleanText)


train, test = train_test_split(df, test_size=0.3, random_state=42)
import nltk
from nltk.corpus import stopwords
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['description']), tags=[r.category]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['description']), tags=[r.category]), axis=1)


#import multiprocessing
#cores = multiprocessing.cpu_count()

model_dbow = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample = 0, workers=4)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)]

%%time
for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressorsdef vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))


##################################
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


## Exapmple document (list of sentences)
doc = ["I love data science",
        "I love coding in python",
        "I love building NLP tool",
        "This is a good phone",
        "This is a good TV",
        "This is a good laptop"]

doc = list(df['description'])
# Tokenization of each document
#stop_words = set(stopwords.words('english'))

from nltk.stem import PorterStemmer
stemming = PorterStemmer()

tokenized_doc = []
tokenized_filtered_doc = []
tokenized_stemmed_doc = []
for d in doc:
    word_tokens = word_tokenize(d.lower())
    #filtered_sentence = [w for w in word_tokens if not w in stop_words]
    #stemmed_sentence = [stemming.stem(word) for word in filtered_sentence]
    #tokenized_doc.append(word_tokenize(d.lower()))
    tokenized_doc.append(word_tokenize(d.lower()))
    #tokenized_filtered_doc.append(filtered_sentence)
    #tokenized_stemmed_doc.append(stemmed_sentence)
#tokenized_doc

# Convert tokenized document into gensim formated tagged data
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
#tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_filtered_doc)]
#tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_stemmed_doc)]
#tagged_data




## Train doc2vec model
model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs = 100)
# Save trained doc2vec model
model.save("test_doc2vec.model")
## Load saved doc2vec model
model= Doc2Vec.load("test_doc2vec.model")
## Print model vocabulary
model.wv.vocab

# find most similar doc
test_doc = word_tokenize("toxin bacteria serum".lower())
model.docvecs.most_similar(positive=[model.infer_vector(test_doc)],topn=5)

doc[2390]
