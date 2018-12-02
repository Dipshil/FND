from preprocessing_utils import *
from liwc_features import LIWC
from tfidf import TFIDFCorpus
from bilstm import BiLSTM
from paths import paths
from tqdm import tqdm
from dnn import DNN
import numpy as np
import io


def run_experiment(model, train, test, epochs):

    model.train(train, epochs)
    train_acc, test_acc = 0.0, 0.0

    for doc, label in train:
        pred = model.predict(doc)
        if pred == label: train_acc += 1

    for doc, label in test:
        pred = model.predict(doc)
        if pred == label: test_acc += 1

    return train_acc/len(train), test_acc/len(test)


def load_vectors(fname):

    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data


def main():

    train_path, dev_path, test_path = paths['TRAIN'], paths['VALID'], paths['TEST']
    train = retrieve_docs(train_path)
    dev = retrieve_docs(dev_path)

    liwc = LIWC()
    train_neg, dev_neg, test_neg = liwc.get_neg_vectors()

    # model = BiLSTM(32, 32, 0.01)
    # acc = run_experiment(model, train, dev, 5)
    # print('Accuracy: %f' % (acc))

    # model = DNN(256, 256, 0.001)
    # acc = run_experiment(model, train, dev, 5)
    # print('Accuracy: %f' % (acc))

    for idx in range(len(train)): train[idx] = (train[idx][0] + [str(freq) for freq in train_neg[idx]], train[idx][1])
    for idx in range(len(dev)): dev[idx] = (dev[idx][0] + [str(freq) for freq in dev_neg[idx]], dev[idx][1])

    tcorp = TFIDFCorpus(train)
    model = DNN(64, 64, 0.005, tcorp)
    train_acc, dev_acc = run_experiment(model, train, dev, 3)
    print('Train Accuracy: %f\tDev Accuracy %f' % (train_acc, dev_acc))


if __name__ == '__main__':
    main()
