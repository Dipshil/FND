from ptrntfidfdnn import PTRNTFIDFDNN
from preprocessing_utils import *
from liwc_features import LIWC
from tfidfdnn import TFIDFDNN
from tfidf import TFIDFCorpus
from ptrndnn import PTRNDNN
from bilstm import BiLSTM
from paths import paths
from lstm import LSTM
from tqdm import tqdm
from dnn import DNN
import numpy as np
import pickle
import io


def transform_data(model, train, dev, test, epochs):

    train_acc, dev_acc, test_acc = run_experiment(model, train, dev, test, epochs)

    transform_train = [(model.transform(doc), label) for doc, label in train]
    transform_dev = [(model.transform(doc), label) for doc, label in dev]
    transform_test = [(model.transform(doc), label) for doc, label in test]

    return train_acc, dev_acc, test_acc, transform_train, transform_dev, transform_test


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

    train_path, dev_path, test_path, out_path = paths['TRAIN'], paths['VALID'], paths['TEST'], './out.csv'
    train, dev, test = retrieve_docs(train_path), retrieve_docs(dev_path), retrieve_docs(test_path)

    ##### TRAIN / TEST MODELS #####

    with open(out_path, 'w') as w: w.write('Model,Train,Dev,Test\n')

    model, name = LSTM(128, 128, 0.005, 5), 'LSTM'
    train_acc, dev_acc, test_acc = run_experiment(model, train, dev, test, 1)
    with open(out_path, 'a') as w: w.write('%s,%f,%f,%f\n' % (name, train_acc, dev_acc, test_acc))

    model, name = BiLSTM(128, 128, 0.005), 'Bidirectional LSTM'
    train_acc, dev_acc, test_acc = run_experiment(model, train, dev, test, 1)
    with open(out_path, 'a') as w: w.write('%s,%f,%f,%f\n' % (name, train_acc, dev_acc, test_acc))

    liwc = LIWC()
    train_all, dev_all, test_all = liwc.get_all_liwc_vectors()
    for idx in range(len(train)): train[idx] = (train[idx][0] + ['ALL_%s_%s'  % (str(ix), str(freq)) for ix, freq in enumerate(train_all[idx])], train[idx][1])
    for idx in range(len(dev)): dev[idx] = (dev[idx][0] + ['ALL_%s_%s' % (str(ix), str(freq)) for ix, freq in enumerate(dev_all[idx])], dev[idx][1])
    for idx in range(len(test)): test[idx] = (test[idx][0] + ['ALL_%s_%s' % (str(ix), str(freq)) for ix, freq in enumerate(test_all[idx])], test[idx][1])
    fast_text = load_vectors(paths['FAST_TEXT'])
    tcorp = TFIDFCorpus(train)

    model, name = DNN(64, 64, 0.005), 'ANN Averaged Embeddings'
    train_acc, dev_acc, test_acc = run_experiment(model, train, dev, test, 1)
    with open(out_path, 'a') as w: w.write('%s,%f,%f,%f\n' % (name, train_acc, dev_acc, test_acc))

    model, name = TFIDFDNN(128, 128, 0.005, tcorp), 'ANN TF-IDF Weighted Embeddings'
    train_acc, dev_acc, test_acc = run_experiment(model, train, dev, test, 1)
    with open(out_path, 'a') as w: w.write('%s,%f,%f,%f\n' % (name, train_acc, dev_acc, test_acc))

    model, name = PTRNDNN(300, 128, 0.005, fast_text), 'ANN Pretrained Embeddings'
    train_acc, dev_acc, test_acc = run_experiment(model, train, dev, test, 1)
    with open(out_path, 'a') as w: w.write('%s,%f,%f,%f\n' % (name, train_acc, dev_acc, test_acc))

    model, name = PTRNTFIDFDNN(300, 128, 0.005, fast_text, tcorp), 'ANN Pretrained TF-IDF Weighted Embeddings'
    train_acc, dev_acc, test_acc = run_experiment(model, train, dev, test, 1)
    with open(out_path, 'a') as w: w.write('%s,%f,%f,%f\n' % (name, train_acc, dev_acc, test_acc))

    ##### TRAIN MODEL AND GENERATE EMBEDDINGS FOR LR/SVM/RFR #####

    liwc = LIWC()
    train_all, dev_all, test_all = liwc.get_all_liwc_vectors()
    for idx in range(len(train)): train[idx] = (train[idx][0] + ['ALL_%s_%s'  % (str(ix), str(freq)) for ix, freq in enumerate(train_all[idx])], train[idx][1])
    for idx in range(len(dev)): dev[idx] = (dev[idx][0] + ['ALL_%s_%s' % (str(ix), str(freq)) for ix, freq in enumerate(dev_all[idx])], dev[idx][1])
    for idx in range(len(test)): test[idx] = (test[idx][0] + ['ALL_%s_%s' % (str(ix), str(freq)) for ix, freq in enumerate(test_all[idx])], test[idx][1])
    tcorp = TFIDFCorpus(train)

    model = TFIDFDNN(128, 128, 0.005, tcorp)
    train_acc, dev_acc, test_acc, transform_train, transform_dev, transform_test = transform_data(model, train, dev, test, 1)
    print('Train Accuracy: %f\tDev Accuracy %f\tTest Accuracy %f' % (train_acc, dev_acc, test_acc))

    with open('transform_train.pickle', 'wb') as w: pickle.dump(transform_train, w)
    with open('transform_dev.pickle', 'wb') as w: pickle.dump(transform_dev, w)
    with open('transform_test.pickle', 'wb') as w: pickle.dump(transform_test, w)


if __name__ == '__main__':
    main()
