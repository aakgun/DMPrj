from models.doc2vec_model import doc2VecModel
from models.classifier_model import classifierModel

import os
import logging
import inspect

import pandas as pd
from sklearn.model_selection import train_test_split

Homedir = os.getcwd()
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
base_file_path = inspect.getframeinfo(inspect.currentframe()).filename
project_dir_path = os.path.dirname(os.path.abspath(base_file_path))
data_path = os.path.join(project_dir_path, 'data')
default_classifier = os.path.join(
    project_dir_path, 'classifiers', 'logreg_model.pkl')
default_doc2vec = os.path.join(project_dir_path, 'classifiers', 'd2v.model')
default_dataset = os.path.join(data_path, 'dataset.csv')


class TextClassifier():

    def __init__(self):
        super().__init__()
        self.d2v = doc2VecModel()
        self.classifier = classifierModel()
        self.dataset = None
        self.dataset1 = None

    def read_data(self, filename):
        filename = os.path.join(data_path, filename)

        # dfdrugbank = pd.read_csv("C:/git/drugbank/data/drugbank.tsv", delimiter="\t")
        # dfprotein = pd.read_csv("C:/git/drugbank/data/proteins.tsv", delimiter="\t")
        # dfdrugbank['DrugBank_id'] = dfdrugbank['drugbank_id']
        # dfprotein['DrugBank_id'] = dfprotein['drugbank_id']
        # dfnodes = pd.read_csv("C:/git/BioNEV/data/DrugBank_DDI/node_list.txt", delimiter="	")
        # dfnodes['ID'] = dfnodes.index
        # dfdrugbank2 = dfdrugbank.merge(dfnodes, how="left", on="DrugBank_id")
        # dfdrugbank2 = dfdrugbank.merge(dfprotein, how="left", on="DrugBank_id")
        # dfdrugbank2.dropna(inplace=True)
        # from sklearn.preprocessing import LabelEncoder
        #
        # lb_make = LabelEncoder()
        # dfdrugbank2["category_code"] = lb_make.fit_transform(dfdrugbank2["category"])

        import pandas as pd

        dfdrugbank = pd.read_csv("C:/git/drugbank/data/drugbank.tsv", delimiter="\t")
        dfprotein = pd.read_csv("C:/git/drugbank/data/proteins.tsv", delimiter="\t")
        dfdrugbank['DrugBank_id'] = dfdrugbank['drugbank_id']
        dfprotein['DrugBank_id'] = dfprotein['drugbank_id']
        dfnodes = pd.read_csv("C:/git/BioNEV/data/DrugBank_DDI/node_list.txt", delimiter="	")
        dfnodes['ID'] = dfnodes.index
        dfnodes['NODEID'] = dfnodes['ID']
        dfnodes['WEIGHT'] = 0
        dfdrugbank2 = dfdrugbank.merge(dfnodes, how="left", on="DrugBank_id")
        dfdrugbank2 = dfdrugbank2.merge(dfprotein, how="left", on="DrugBank_id")
        # dfdrugbank3 = dfdrugbank2[str(dfdrugbank2['ID'])!="nan"]
        dfdrugbank2.dropna(inplace=True)
        from sklearn.preprocessing import LabelEncoder
        lb_make = LabelEncoder()
        dfdrugbank2["category_code"] = lb_make.fit_transform(dfdrugbank2["category"])
        dfdrugbank2["actions_code"] = lb_make.fit_transform(dfdrugbank2["actions"])
        dfdrugbank2["organism_code"] = lb_make.fit_transform(dfdrugbank2["organism"])
        dfdrugbank2["groups_code"] = lb_make.fit_transform(dfdrugbank2["groups"])

        dfdrugbank2['ID'] = dfdrugbank2['ID'].astype(int)


        #print(dfdrugbank2.head(5))
        #print(dfdrugbank2.columns)

        self.dataset = dfdrugbank2
        #tc.dataset = dfdrugbank2
        #filename="C:/git/doc2vec/data/dataset.csv"
        #self.dataset = pd.read_csv(filename, header=0, delimiter="\t")


    def prepare_all_data(self):
        #x_train, x_test, y_train, y_test = train_test_split(
        #    self.dataset.review, self.dataset.sentiment, random_state=0,
        #    test_size=0.1)
        x_train, x_test, y_train, y_test = train_test_split(
            self.dataset.description, self.dataset.organism_code, random_state=0,
            test_size=0.1)
        x_train = doc2VecModel.label_sentences(x_train, 'Train')
        x_test = doc2VecModel.label_sentences(x_test, 'Test')
        #x_train2 = doc2VecModel.label_sentences2(x_train, 'Train')
        all_data = x_train + x_test
        return x_train, x_test, y_train, y_test, all_data

        #x_train, x_test, y_train, y_test = train_test_split(tc.dataset.review, tc.dataset.sentiment, random_state=0,test_size=0.1)
        #x_train, x_test, y_train, y_test = train_test_split(tc.dataset.description, tc.dataset.category_code, random_state=0,test_size=0.1)
        #x_train = doc2VecModel.label_sentences(x_train, 'Train')

        x_train2 = doc2VecModel.label_sentences2(x_train, 'Train')
        x_train2 = label_sentences2(x_train, 'Train')

    def prepare_test_data(self, sentence):
        x_test = doc2VecModel.label_sentences(sentence, 'Test')
        return x_test

    def train_classifier(self):
        x_train, x_test, y_train, y_test, all_data = self.prepare_all_data()
        self.d2v.initialize_model(all_data)
        self.d2v.train_model()
        self.classifier.initialize_model()
        self.classifier.train_model(self.d2v, x_train, y_train)
        self.classifier.test_model(self.d2v, x_test, y_test)
        return self.d2v, self.classifier

    def test_classifier(self):
        _, x_test, _, y_test, _ = self.prepare_all_data()
        if (self.d2v.model is None or self.classifier.model is None):
            logging.info(
                "Models Not Found, Train First or Use Correct Model Names")
        else:
            self.classifier.test_model(self.d2v, x_test, y_test)

#
# def run(dataset_file):
#     tc = TextClassifier()
#     tc.read_data(dataset_file)
#     tc.train_classifier()
#     tc.test_classifier()
#
#
# if __name__ == "__main__":
#     run("dataset.csv")
#
# run("dataset.csv")
# tc = TextClassifier()
# ds = tc.dataset
#
# tc.read_data("C:/git/doc2vec/data/dataset.csv")
# tc.train_classifier()
# x_train, x_test, y_train, y_test, all_data = tc.prepare_all_data()
# #############################################################
# x_train, x_test, y_train, y_test = train_test_split(
#     tc.dataset.description, tc.dataset.actions_code, random_state=0,
#     test_size=0.1)
# x_train = doc2VecModel.label_sentences(x_train, 'Train')
# x_test = doc2VecModel.label_sentences(x_test, 'Test')
# all_data = x_train + x_test
# ################################################################
# x_train = doc2VecModel.label_sentences(x_train, 'Train')
# x_test = doc2VecModel.label_sentences(x_test, 'Test')
# all_data = x_train + x_test
#
# d2v.initialize_model(all_data)
# d2v.train_model()
# classifier.initialize_model()
# classifier.train_model(self.d2v, x_train, y_train)
# classifier.test_model(self.d2v, x_test, y_test)
# return self.d2v, self.classifier
#
#
# tc.train_classifier()