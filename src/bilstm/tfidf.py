from tqdm import tqdm
import numpy as np


class TFIDFCorpus():


    def __init__(self, docs):

        self.tf, self.df, self.N = {}, {}, 0
        for tokens, labels in tqdm(docs):

            for token in tokens:
                if token in self.tf: self.tf[token] += 1
                else: self.tf[token] = 1

            unique = set(tokens)

            for token in unique:
                if token in self.df: self.df[token] += 1
                else: self.df[token] = 1

            self.N += 1


    def get_tf(self, term):

        if term in self.tf: return self.tf[term]
        else: return 0


    def get_df(self, term):

        if term in self.df: return self.df[term]
        else: return 0


    def get_tf_idf(self, term):

        tf = np.log(1 + self.get_tf(term))

        frac = (self.N / self.get_df(term)) if self.get_df(term) != 0 else 0
        idf = np.log(1 + frac)

        return tf * idf


    def get_tf_idf_seq(self, seq):

        return [self.get_tf_idf(term) for term in seq]
