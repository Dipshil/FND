from collections import Counter, defaultdict

import numpy as np

from paths import *
from tfidf_corpus import TF_IDF_Corpus


class LIWC:
    def __init__(self, ngram_limit=2, min_df=2, is_sublinear_tf=True):
        self.tfidf_corp = TF_IDF_Corpus(
            ngram_limit=2, min_df=2, is_sublinear_tf=True)

    def get_labels(self, file_path):
        with open(file_path, 'r') as f:
            labels = []
            lines = f.readlines()
            for line in lines:
                line = line.strip().split("\t")
                labels.append(line[1])
        return labels

    def parse_liwc_results(self, file_path, gram=False, neg=False):
        categories = defaultdict(dict)
        with open(file_path, 'r') as f:
            if gram:
                cat_list = f.readline().strip().split("\t")[1:22]
            elif neg:
                cat_list = f.readline().strip().split("\t")[24:28]
            else:
                cat_list = f.readline().strip().split("\t")

            for line in f:
                line = line.strip().split("\t")
                token = line[0]

                if gram:
                    occurences = np.where(np.asarray(line[1:22]) == 'X')[0]
                elif neg:
                    occurences = np.where(np.asarray(line[24:28]) == 'X')[0]
                else:
                    occurences = np.where(np.asarray(line[1:]) == 'X')[0]
                cats = np.asarray(cat_list)[occurences]
                for cat in cats:
                    categories[token][cat] = cat_list.index(cat)
        return categories, cat_list

    def vectorize(self, file_path, categories, cat_list):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            vector = np.zeros((len(lines), len(cat_list)))
            for doc_idx, line in enumerate(lines):
                token_counts = Counter(line.strip().split())
                for tok_idx, token in enumerate(token_counts):
                    cats = categories[token]
                    for cat_idx in cats.values():
                        vector[doc_idx][cat_idx] = token_counts[token]
        return vector

    def gram_vectorize(self, corpus_path, liwc_path):
        categories, cat_list = self.parse_liwc_results(liwc_path, gram=True)

        return self.vectorize(corpus_path, categories, cat_list)

    def neg_vectorize(self, corpus_path, liwc_path):
        categories, cat_list = self.parse_liwc_results(liwc_path, neg=True)

        return self.vectorize(corpus_path, categories, cat_list)

    def get_tfidf_vectors(self):
        tfidf_train = self.tfidf_corp.tfidf_train_vectors
        tfidf_valid = self.tfidf_corp.tfidf_valid_vectors
        tfidf_test = self.tfidf_corp.tfidf_test_vectors

        return tfidf_train, tfidf_valid, tfidf_test

    def get_gram_vectors(self):
        train_gram = self.gram_vectorize(TRAIN_PATH, TRAIN_RES_PATH)
        valid_gram = self.gram_vectorize(VALID_PATH, VALID_RES_PATH)
        test_gram = self.gram_vectorize(TEST_PATH, TEST_RES_PATH)

        return train_gram, valid_gram, test_gram

    def get_neg_vectors(self):
        train_neg = self.neg_vectorize(TRAIN_PATH, TRAIN_RES_PATH)
        valid_neg = self.neg_vectorize(VALID_PATH, VALID_RES_PATH)
        test_neg = self.neg_vectorize(TEST_PATH, TEST_RES_PATH)

        return train_neg, valid_neg, test_neg

    def get_all_liwc_vectors(self):
        train = self.all_vectorize(TRAIN_PATH, TRAIN_RES_PATH)
        valid = self.all_vectorize(VALID_PATH, VALID_RES_PATH)
        test = self.all_vectorize(TEST_PATH, TEST_RES_PATH)

        return train, valid, test

    def all_vectorize(self, corpus_path, liwc_path):
        categories, cat_list = self.parse_liwc_results(liwc_path)
        return self.vectorize(corpus_path, categories, cat_list)

    def get_deep_tfidf_vectors(self):
        train_vectors, valid_vectors, test_vectors = [], [], []
        with open(TRAIN_DEEP_PATH, 'r') as f:
            next(f)
            for line in f:
                line = line.strip().split(",")
                train_vectors.append(np.array(list(map(float, line))))
        with open(VALID_DEEP_PATH, 'r') as f:
            next(f)
            for line in f:
                line = line.strip().split(",")
                valid_vectors.append(np.array(list(map(float, line))))
        with open(TEST_DEEP_PATH, 'r') as f:
            next(f)
            for line in f:
                line = line.strip().split(",")
                test_vectors.append(np.array(list(map(float, line))))

        return np.array(train_vectors), np.array(valid_vectors), np.array(test_vectors)
