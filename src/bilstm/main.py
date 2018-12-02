from ptrntfidfdnn import PTRNTFIDFDNN
from preprocessing_utils import *
from liwc_features import LIWC
from tfidfdnn import TFIDFDNN
from tfidf import TFIDFCorpus
from ptrndnn import PTRNDNN
from bilstm import BiLSTM
from paths import paths
from tqdm import tqdm
from dnn import DNN
import numpy as np
import io


def run_experiment(model, train, dev, test, epochs):

    model.train(train, epochs)
    train_acc, dev_acc, test_acc = 0.0, 0.0, 0.0

    for doc, label in train:
        pred = model.predict(doc)
        if pred == label: train_acc += 1

    for doc, label in dev:
        pred = model.predict(doc)
        if pred == label: dev_acc += 1

    for doc, label in test:
        pred = model.predict(doc)
        if pred == label: test_acc += 1

    return train_acc/len(train), dev_acc/len(dev), test_acc/len(test)


def load_vectors(fname, lim=None):

    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    data, count = {}, 0
    for line in fin:

        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))

        if lim != None:
            if count >= lim: break

        count += 1

    return data


def main():

    train_path, dev_path, test_path = paths['TRAIN'], paths['VALID'], paths['TEST']
    train = retrieve_docs(train_path)
    dev = retrieve_docs(dev_path)
    test = retrieve_docs(test_path)

    liwc = LIWC()
    train_gram, dev_gram, test_gram = liwc.get_gram_vectors()
    train_neg, dev_neg, test_neg = liwc.get_neg_vectors()
    train_all, dev_all, test_all = liwc.get_all_liwc_vectors()

    # model = BiLSTM(32, 32, 0.01)
    # acc = run_experiment(model, train, dev, 5)
    # print('Accuracy: %f' % (acc))

    # model = DNN(256, 256, 0.001)
    # acc = run_experiment(model, train, dev, 5)
    # print('Accuracy: %f' % (acc))

    # for idx in range(len(train)): train[idx] = (train[idx][0] + ['GRAM_%s_%s'  % (str(ix), str(freq)) for ix, freq in enumerate(train_gram[idx])], train[idx][1])
    # for idx in range(len(dev)): dev[idx] = (dev[idx][0] + ['GRAM_%s_%s' % (str(ix), str(freq)) for ix, freq in enumerate(dev_gram[idx])], dev[idx][1])
    # for idx in range(len(test)): test[idx] = (test[idx][0] + ['GRAM_%s_%s' % (str(ix), str(freq)) for ix, freq in enumerate(test_gram[idx])], test[idx][1])
    # for idx in range(len(train)): train[idx] = (train[idx][0] + ['NEG_%s_%s' % (str(ix), str(freq)) for ix, freq in enumerate(train_neg[idx])], train[idx][1])
    # for idx in range(len(dev)): dev[idx] = (dev[idx][0] + ['NEG_%s_%s' % (str(ix), str(freq)) for ix, freq in enumerate(dev_neg[idx])], dev[idx][1])
    # for idx in range(len(test)): test[idx] = (test[idx][0] + ['NEG_%s_%s' % (str(ix), str(freq)) for ix, freq in enumerate(test_neg[idx])], test[idx][1])
    for idx in range(len(train)): train[idx] = (train[idx][0] + ['ALL_%s_%s'  % (str(ix), str(freq)) for ix, freq in enumerate(train_all[idx])], train[idx][1])
    for idx in range(len(dev)): dev[idx] = (dev[idx][0] + ['ALL_%s_%s' % (str(ix), str(freq)) for ix, freq in enumerate(dev_all[idx])], dev[idx][1])
    for idx in range(len(test)): test[idx] = (test[idx][0] + ['ALL_%s_%s' % (str(ix), str(freq)) for ix, freq in enumerate(test_all[idx])], test[idx][1])

    # model = DNN(64, 64, 0.005)
    # train_acc, dev_acc, test_acc = run_experiment(model, train, dev, 1)
    # print('Train Accuracy: %f\tDev Accuracy %f\tTest Accuracy %f' % (train_acc, dev_acc, test_acc))

    tcorp = TFIDFCorpus(train)
    model = TFIDFDNN(128, 128, 0.0005, tcorp)
    # model = TFIDFDNN(128, 128, 0.005, tcorp)
    train_acc, dev_acc, test_acc = run_experiment(model, train, dev, test, 1)
    print('Train Accuracy: %f\tDev Accuracy %f\tTest Accuracy %f' % (train_acc, dev_acc, test_acc))

    # fast_text = load_vectors(paths['FAST_TEXT'])
    # model = PTRNDNN(300, 128, 0.005, fast_text)
    # train_acc, dev_acc, test_acc = run_experiment(model, train, dev, test, 1)
    # print('Train Accuracy: %f\tDev Accuracy %f\tTest Accuracy %f' % (train_acc, dev_acc, test_acc))

    # tcorp = TFIDFCorpus(train)
    # fast_text = load_vectors(paths['FAST_TEXT'])
    # model = PTRNTFIDFDNN(300, 128, 0.005, fast_text, tcorp)
    # train_acc, dev_acc, test_acc = run_experiment(model, train, dev, test, 1)
    # print('Train Accuracy: %f\tDev Accuracy %f\tTest Accuracy %f' % (train_acc, dev_acc, test_acc))


if __name__ == '__main__':
    main()
