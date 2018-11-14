TRAIN_PATH = "./liar_dataset/train.tsv"
VALID_PATH = "./liar_dataset/valid.tsv"
TEST_PATH = "./liar_dataset/test.tsv"

from sklearn.feature_extraction.text import TfidfVectorizer


class TF_IDF_Corpus:
    def __init__(self, ngram_limit=2, min_df=2, is_sublinear_tf=True):

        train_data, self.train_labels = self.read_data(TRAIN_PATH)
        valid_data, self.valid_labels = self.read_data(VALID_PATH)
        test_data, self.test_labels = self.read_data(TEST_PATH)

        self.vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', ngram_range=(
            1, ngram_limit), min_df=min_df, sublinear_tf=is_sublinear_tf)

        self.train_vectors = self.vectorizer.fit_transform(train_data)
        self.valid_vectors = self.vectorizer.transform(valid_data)
        self.test_vectors = self.vectorizer.transform(test_data)

    def read_data(self, file_path):
        data, labels = [], []
        with open(file_path) as f:
            for line in f:
                line = line.strip('\n').split('\t')
                labels.append(line[1])
                data.append(line[2])
        return data, labels
