import torch.nn.functional as F
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch


class PTRNDNN(nn.Module):


    def __init__(self, embedding_dim, hidden_dim, lr, pt):
        super(PTRNDNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.pt = pt

        self.word_to_ix = {}
        self.label_to_ix = {}


    def forward(self, sentence):

        lin1_out = self.lin1(sentence.view(1, 1, -1)).cuda()
        lin2_out = self.lin2(lin1_out.view(1, 1, -1)).cuda()
        lin3_out = self.lin3(lin2_out.view(1, 1, -1)).cuda()
        lin4_out = self.lin4(lin3_out.view(1, 1, -1)).cuda()
        lin5_out = self.lin5(lin4_out.view(1, 1, -1)).cuda()
        label_space = self.hidden2tag(lin5_out.view(1, -1)).cuda()
        label_scores = F.log_softmax(label_space, dim=1).cuda()

        return label_scores


    def init(self, docs):

        self.init_vocab(docs)
        self.lin1 = nn.Linear(self.embedding_dim, self.hidden_dim).cuda()
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()
        self.lin3 = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()
        self.lin4 = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()
        self.lin5 = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()
        self.hidden2tag = nn.Linear(self.hidden_dim, len(self.label_to_ix)).cuda()
        self.loss_function = nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adagrad(self.parameters(), lr=self.lr)


    def init_vocab(self, docs):

        for doc, label in docs:

            for word in doc:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)

            if label not in self.label_to_ix:
                self.label_to_ix[label] = len(self.label_to_ix)

        self.word_to_ix['UNK'] = len(self.word_to_ix)

        self.ix_to_label = { self.label_to_ix[label]: label for label in self.label_to_ix.keys() }
        self.ix_to_word = { self.word_to_ix[word]: word for word in self.word_to_ix.keys() }


    def train(self, docs, epochs):

        self.init(docs)

        for epoch in trange(epochs):
            timestep = tqdm(docs, desc='loss')
            for doc, label in timestep:

                self.zero_grad()

                sentence_idxs = [self.word_to_ix[w] for w in doc]
                sentence_in = torch.tensor(sentence_idxs, dtype=torch.long).cuda()
                print(sentence_in.view(1, 1, -1).size())

                sentence_idxs = [self.pt[w] for w in doc if w in self.pt]
                sentence_in = torch.tensor(sentence_idxs, dtype=torch.long).cuda()
                print(sentence_in.view(1, 1, -1).size())
                exit()

                target_idxs = [self.label_to_ix[label]]
                targets = torch.tensor(target_idxs, dtype=torch.long).cuda()

                label_scores = self(sentence_in)

                loss = self.loss_function(label_scores, targets)

                timestep.set_description('loss=%.4f' % loss)
                timestep.refresh()

                loss.backward()
                self.optimizer.step()

        print()


    def predict(self, doc):

        with torch.no_grad():

            idxs = [self.word_to_ix[w] if w in self.word_to_ix else self.word_to_ix['UNK'] for w in doc]
            inputs = torch.tensor(idxs, dtype=torch.long).cuda()
            label_scores = self(inputs)
            lix = np.argmax(label_scores[-1]).item()

            return self.ix_to_label[lix]
