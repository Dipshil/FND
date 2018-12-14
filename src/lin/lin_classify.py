from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from scipy.stats import ttest_ind, f_oneway
from sklearn.metrics import confusion_matrix
import itertools

from liwc_features import LIWC
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from paths import *

np.random.seed(23753)


class LinModel():

    def __init__(self):
        self.svm = LinearSVC(max_iter=10000)
        self.lr = LogisticRegression(
            multi_class='auto', solver='liblinear', n_jobs=-1)
        self.rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        self.liwc = LIWC()

    def classify(self):

        train, valid, test = self.liwc.get_tfidf_vectors()
        train2, valid2, test2 = self.liwc.get_all_liwc_vectors()
        #train3, valid3, test3 = self.liwc.get_deep_tfidf_vectors()

        # train_gram, valid_gram, _ = self.liwc.get_gram_vectors()
        # train_neg, valid_neg, _ = self.liwc.get_neg_vectors()

        train_vector = np.concatenate(
            [train.toarray(), train2], axis=1)
        valid_vector = np.concatenate(
            [valid.toarray(), valid2], axis=1)
        test_vector = np.concatenate(
            [test.toarray(), test2], axis=1)

        train_labels = self.liwc.get_labels(TRAIN_PATH)
        valid_labels = self.liwc.get_labels(VALID_PATH)
        test_labels = self.liwc.get_labels(TEST_PATH)

        self.svm.fit(train_vector, train_labels)
        self.lr.fit(train_vector, train_labels)
        self.rfc.fit(train_vector, train_labels)

        print("TRAIN\n")
        pred = self.svm.predict(train_vector)
        f1 = f1_score(train_labels, pred, average='weighted')
        acc = accuracy_score(train_labels, pred)
        print(f1, acc)

        pred = self.lr.predict(train_vector)
        f1 = f1_score(train_labels, pred, average='weighted')
        acc = accuracy_score(train_labels, pred)
        print(f1, acc)

        pred = self.rfc.predict(train_vector)
        f1 = f1_score(train_labels, pred, average='weighted')
        acc = accuracy_score(train_labels, pred)
        print(f1, acc)

        print("VALIDATION\n")
        pred = self.svm.predict(valid_vector)
        f1 = f1_score(valid_labels, pred, average='weighted')
        acc = accuracy_score(valid_labels, pred)
        print(f1, acc)

        pred = self.lr.predict(valid_vector)
        f1 = f1_score(valid_labels, pred, average='weighted')
        acc = accuracy_score(valid_labels, pred)
        print(f1, acc)

        pred = self.rfc.predict(valid_vector)
        f1 = f1_score(valid_labels, pred, average='weighted')
        acc = accuracy_score(valid_labels, pred)
        print(f1, acc)

        print("\nTESTING\n")

        pred_svm = self.svm.predict(test_vector)
        f1 = f1_score(test_labels, pred_svm, average='weighted')
        acc = accuracy_score(test_labels, pred_svm)
        print(f1, acc)

        pred_lr = self.lr.predict(test_vector)
        f1 = f1_score(test_labels, pred_lr, average='weighted')
        acc_lr = accuracy_score(test_labels, pred_lr)
        print(f1, acc_lr)

        pred_rfc = self.rfc.predict(test_vector)
        f1 = f1_score(test_labels, pred_rfc, average='weighted')
        acc = accuracy_score(test_labels, pred_rfc)
        print(f1, acc)

        '''labels = set(train_labels)
        label_ind = {label: i for i, label in enumerate(labels)}
        pred_svm = list(map(lambda x: label_ind[x], pred_svm))
        pred_lr = list(map(lambda x: label_ind[x], pred_lr))
        pred_rfc = list(map(lambda x: label_ind[x], pred_rfc))
        print(ttest_ind(pred_svm, pred_lr))
        print(ttest_ind(pred_lr, pred_rfc))
        print(ttest_ind(pred_svm, pred_rfc))
        print(f_oneway(pred_svm, pred_lr, pred_rfc))'''

        self.feat_importance(train_vector)

        classes = []
        present = {}
        for idx, item in enumerate(test_labels):
            if not present.get(item):
                present[item] = idx + 1
                classes.append(item)

        '''pred_mt = [pred for pred in pred_lr if pred == "mostly-true"]
        true_mt = [pred for pred in test_labels if pred == "mostly-true"]
        print(len(pred_mt), len(true_mt))
        cm = confusion_matrix(test_labels, pred_lr)
        plt.figure(figsize=(10, 10))
        plot_confusion_matrix(cm, classes, normalize=True)'''

    def feat_importance(self, train_vector):
        feat_imp = self.rfc.feature_importances_
        liwc_cat_list = self.liwc.cat_list
        tfidf_vocab = self.liwc.tfidf_corp.vocab
        features = list(tfidf_vocab.keys()) + liwc_cat_list
        feat_importances = pd.Series(
            self.rfc.feature_importances_, index=features)
        feat_importances.nlargest(5).plot(
            kind='barh', figsize=(20, 20), fontsize=20)
        plt.savefig("features.png")


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # From sklearn documentation
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    plt.savefig("cm.png")
    plt.tight_layout()


if __name__ == "__main__":
    model = LinModel()
    model.classify()
