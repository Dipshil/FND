from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from scipy.stats import ttest_ind
from liwc_features import LIWC
from paths import *
import numpy as np
import pickle


class LinModel():

    def __init__(self):

        self.svm = LinearSVC(max_iter=10000)
        self.lr = LogisticRegression(multi_class='auto', solver='liblinear')
        self.rfc = RandomForestClassifier(n_estimators=100)
        self.liwc = LIWC()


    def classify(self):

        with open('./transform_train.pickle', 'rb') as r: train = pickle.load(r)
        with open('./transform_dev.pickle', 'rb') as r: dev = pickle.load(r)
        with open('./transform_test.pickle', 'rb') as r: test = pickle.load(r)

        train_docs, train_labels = zip(*train)
        dev_docs, dev_labels = zip(*dev)
        test_docs, test_labels = zip(*test)

        self.svm.fit(train_docs, train_labels)
        # self.lr.fit(train_docs, train_labels)
        # self.rfc.fit(train_docs, train_labels)

        print("TRAINING\n")
        pred = self.svm.predict(train_docs)
        acc = accuracy_score(train_labels, pred)
        print(acc)

        # pred = self.lr.predict(train_docs)
        # acc = accuracy_score(train_labels, pred)
        # print(acc)

        # pred = self.rfc.predict(train_docs)
        # acc = accuracy_score(train_labels, pred)
        # print(acc)

        print("VALIDATION\n")
        pred = self.svm.predict(dev_docs)
        acc = accuracy_score(dev_labels, pred)
        print(acc)

        # pred = self.lr.predict(dev_docs)
        # acc = accuracy_score(dev_labels, pred)
        # print(acc)

        # pred = self.rfc.predict(dev_docs)
        # acc = accuracy_score(dev_labels, pred)
        # print(acc)

        print("\nTESTING\n")
        pred_svm = self.svm.predict(test_docs)
        acc = accuracy_score(test_labels, pred_svm)
        print(acc)

        # pred = self.lr.predict(test_docs)
        # acc = accuracy_score(test_labels, pred)
        # print(acc)

        # pred = self.rfc.predict(test_docs)
        # acc = accuracy_score(test_labels, pred)
        # print(acc)

        # labels = set(train_labels)
        # label_ind = {label: i for i, label in enumerate(labels)}
        # pred_svm = list(map(lambda x: label_ind[x], pred_svm))
        # pred_lr = list(map(lambda x: label_ind[x], pred_lr))
        # print(ttest_ind(pred_svm, pred_lr))


if __name__ == "__main__":
    model = LinModel()
    model.classify()
