from preprocessing_utils import *
from liwc_features import LIWC
from bilstm import BiLSTM
from paths import paths


def run_experiment(model, train, test, epochs):

    model.train(train, epochs)
    acc = 0.0

    for doc, label in test:
        pred = model.predict(doc)
        if pred == label: acc += 1

    acc /= len(test)
    return acc


def main():

    train_path, dev_path, test_path = paths['TRAIN'], paths['VALID'], paths['TEST']
    train = retrieve_docs(train_path)
    dev = retrieve_docs(dev_path)

    liwc = LIWC()

    # model = BiLSTM(32, 32, 0.01)
    # acc = run_experiment(model, train, dev, 5)

    # print('Accuracy: %f' % (acc))
    # print(len(model.word_to_ix))

if __name__ == '__main__':
    main()
