import torch.nn.functional as F
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch


class BiLSTM(nn.Module):


    def __init__(self, embedding_dim, hidden_dim, lr):

        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.word_to_ix = {}
        self.label_to_ix = {}
        self.lr = lr


    def init_hidden(self):

        return (torch.zeros(2, 1, self.hidden_dim // 2).cuda(), torch.zeros(2, 1, self.hidden_dim // 2).cuda())


    def forward(self, sentence):

        embeds = self.word_embeddings(sentence).cuda()
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        label_space = self.hidden2tag(lstm_out.view(len(sentence), -1)).cuda()
        label_scores = F.log_softmax(label_space, dim=1).cuda()

        return label_scores


    def init(self, docs):

        self.init_vocab(docs)
        self.init_model()
        self.loss_function = nn.NLLLoss().cuda()
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)


    def init_vocab(self, docs):

        for doc, label in docs:

            for word in doc:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)

            if label not in self.label_to_ix:
                self.label_to_ix[label] = len(self.label_to_ix)

        self.word_to_ix['UNK'] = len(self.word_to_ix)
        self.ix_to_label = { self.label_to_ix[label]: label for label in self.label_to_ix.keys() }


    def init_model(self):

        self.word_embeddings = nn.Embedding(len(self.word_to_ix), self.embedding_dim).cuda()
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True).cuda()
        self.hidden2tag = nn.Linear(self.hidden_dim, len(self.label_to_ix)).cuda()
        self.hidden = self.init_hidden()


    def train(self, docs, epochs):

        self.init(docs)

        for epoch in trange(epochs):
            for doc, label in tqdm(docs):

                self.zero_grad()
                self.hidden = self.init_hidden()

                sentence_idxs = [self.word_to_ix[w] for w in doc]
                sentence_in = torch.tensor(sentence_idxs, dtype=torch.long).cuda()

                target_idxs = [self.label_to_ix[label] for w in doc]
                targets = torch.tensor(target_idxs, dtype=torch.long).cuda()

                label_scores = self(sentence_in)

                loss = self.loss_function(label_scores, targets)
                loss.backward()
                self.optimizer.step()


    def predict(self, doc):

        with torch.no_grad():

            idxs = [self.word_to_ix[w] if w in self.word_to_ix else self.word_to_ix['UNK'] for w in doc]
            inputs = torch.tensor(idxs, dtype=torch.long).cuda()
            label_scores = self(inputs)
            lix = np.argmax(label_scores[-1]).item()

            return self.ix_to_label[lix]
