import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
import torch.nn as nn
import torch


class BiLSTM(nn.Module):


    def __init__(self, embedding_dim, hidden_dim):

        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.word_to_ix = {}
        self.label_to_ix = {}


    def init_hidden(self):

        return (torch.zeros(2, 1, self.hidden_dim // 2), torch.zeros(2, 1, self.hidden_dim // 2))


    def forward(self, sentence):

        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        label_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        label_scores = F.log_softmax(label_space, dim=1)

        return label_scores


    def init(self, docs):

        self.init_vocab(docs)
        self.init_model()
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.1)


    def init_vocab(self, docs):

        for doc, label in docs:

            for word in doc:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)

            if label not in self.label_to_ix:
                self.label_to_ix[label] = len(self.label_to_ix)


    def init_model(self):

        self.word_embeddings = nn.Embedding(len(self.word_to_ix), self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, len(self.label_to_ix))
        self.hidden = self.init_hidden()


    def train(self, docs, epochs):

        self.init(docs)

        for epoch in trange(epochs):
            for doc, label in docs:

                self.zero_grad()
                self.hidden = self.init_hidden()

                sentence_idxs = [self.word_to_ix[w] for w in doc]
                sentence_in = torch.tensor(sentence_idxs, dtype=torch.long)

                target_idxs = [self.label_to_ix[label] for w in doc]
                targets = torch.tensor(target_idxs, dtype=torch.long)

                label_scores = self(sentence_in)

                loss = self.loss_function(label_scores, targets)
                loss.backward()
                self.optimizer.step()


    def predict(self, doc):

        with torch.no_grad():

            idxs = [self.word_to_ix[w] if w in self.word_to_ix else self.word_to_ix['UNK'] for w in doc]
            inputs = torch.tensor(idxs, dtype=torch.long)
            label_scores = self(inputs)

            return label_scores[-1]
