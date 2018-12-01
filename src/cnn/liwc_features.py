from tfidf_corpus import TF_IDF_Corpus
from sklearn.feature_extraction.text import CountVectorizer
from paths import *


class LIWC:
    def __init__(self, ngram_limit=2, min_df=2, is_sublinear_tf=True):
        self.tfidf_corp = TF_IDF_Corpus(
            ngram_limit=2, min_df=2, is_sublinear_tf=True)
        self.vocab = self.tfidf_corp.vocab

        self.count_vectorizer = CountVectorizer(
            min_df=2, vocabulary=self.vocab)

        train_counts = self.count_vectorizer.fit_transform(
            self.tfidf_corp.train_data)
        valid_counts = self.count_vectorizer.transform(
            self.tfidf_corp.valid_data)
        test_counts = self.count_vectorizer.transform(
            self.tfidf_corp.test_data)




            
