from sklearn.feature_extraction.text import TfidfVectorizer
from paths import *


class TF_IDF_Corpus:


    def __init__(self, ngram_limit=2, min_df=2, is_sublinear_tf=True):

        self.train_data, self.train_labels = self.read_data(paths['TRAIN'])
        self.valid_data, self.valid_labels = self.read_data(paths['VALID'])
        self.test_data, self.test_labels = self.read_data(paths['TEST'])

        self.vectorizer = TfidfVectorizer(ngram_range=(1, ngram_limit), min_df=min_df, sublinear_tf=is_sublinear_tf)

        self.tfidf_train_vectors = self.vectorizer.fit_transform(self.train_data)
        self.tfidf_valid_vectors = self.vectorizer.transform(self.valid_data)
        self.tfidf_test_vectors = self.vectorizer.transform(self.test_data)

        self.vocab = self.vectorizer.vocabulary_


    def read_data(self, file_path):

        data, labels = [], []
        with open(file_path) as f:

            for line in f:
                line = line.strip('\n').split('\t')
                labels.append(line[1])
                data.append(line[2])

        return data, labels
