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
        self.lr = LogisticRegression(multi_class='ovr', solver='liblinear')
        self.rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        self.liwc = LIWC()


    def classify(self):

        with open('./transform_train.pickle', 'rb') as r: train = pickle.load(r)
        with open('./transform_dev.pickle', 'rb') as r: dev = pickle.load(r)
        with open('./transform_test.pickle', 'rb') as r: test = pickle.load(r)

        train_docs, train_labels = zip(*train)
        dev_docs, dev_labels = zip(*dev)
        test_docs, test_labels = zip(*test)

        self.svm.fit(train_docs, train_labels)
        self.lr.fit(train_docs, train_labels)
        self.rfc.fit(train_docs, train_labels)

        pred = self.svm.predict(train_docs)
        svm_train_acc = accuracy_score(train_labels, pred)

        pred = self.lr.predict(train_docs)
        lr_train_acc = accuracy_score(train_labels, pred)

        pred = self.rfc.predict(train_docs)
        rfc_train_acc = accuracy_score(train_labels, pred)

        pred = self.svm.predict(dev_docs)
        svm_dev_acc = accuracy_score(dev_labels, pred)

        pred = self.lr.predict(dev_docs)
        lr_dev_acc = accuracy_score(dev_labels, pred)

        pred = self.rfc.predict(dev_docs)
        rfc_dev_acc = accuracy_score(dev_labels, pred)

        pred_svm = self.svm.predict(test_docs)
        svm_test_acc = accuracy_score(test_labels, pred_svm)

        pred = self.lr.predict(test_docs)
        lr_test_acc = accuracy_score(test_labels, pred)

        pred = self.rfc.predict(test_docs)
        rfc_test_acc = accuracy_score(test_labels, pred)

        with open('./out.csv', 'a') as a:
            a.write('%s,%f,%f,%f\n' % ('SVM', svm_train_acc, svm_dev_acc, svm_test_acc))
            a.write('%s,%f,%f,%f\n' % ('LR', lr_train_acc, lr_dev_acc, lr_test_acc))
            a.write('%s,%f,%f,%f\n' % ('RFC', rfc_train_acc, rfc_dev_acc, rfc_test_acc))

if __name__ == "__main__":
    model = LinModel()
    model.classify()
