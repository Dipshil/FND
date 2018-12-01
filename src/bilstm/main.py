from preprocessing_utils import *
from bilstm import BiLSTM
from paths import paths

EMBEDDING_DIM = 6
HIDDEN_DIM = 6 


def main():

    train, valid, test = paths['TRAIN'], paths['VALID'], paths['TEST']
    train_docs = retrieve_docs(train)[:10]
    valid_docs = retrieve_docs(valid)[:10]

    model = BiLSTM(EMBEDDING_DIM, HIDDEN_DIM)
    model.train(train_docs, 300)
    print(model.predict(valid_docs[0][0]))


if __name__ == '__main__':
    main()
