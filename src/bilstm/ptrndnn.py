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

        embeds = torch.mean(sentence, dim=0).cuda()
        lin1_out = self.lin1(embeds).cuda()
        d1_out = self.drop(lin1_out).cuda()
        lin2_out = self.lin2(d1_out).cuda()
        d2_out = self.drop(lin2_out).cuda()
        lin3_out = self.lin3(d2_out).cuda()
        lin4_out = self.lin4(lin3_out).cuda()
        lin5_out = self.lin5(lin4_out).cuda()
        label_space = self.hidden2tag(lin5_out).cuda()

        return label_space.view(1, -1)


    def init(self, docs):

        self.init_vocab(docs)
        self.lin1 = nn.Linear(self.embedding_dim, self.hidden_dim).cuda()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin3 = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()
        torch.nn.init.xavier_uniform_(self.lin3.weight)
        self.lin4 = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()
        torch.nn.init.xavier_uniform_(self.lin4.weight)
        self.lin5 = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()
        torch.nn.init.xavier_uniform_(self.lin5.weight)
        self.hidden2tag = nn.Linear(self.hidden_dim, len(self.label_to_ix)).cuda()
        torch.nn.init.xavier_uniform_(self.hidden2tag.weight)
        self.drop = nn.Dropout(0.2)
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

                sentence_idxs = [self.pt[w] for w in doc if w in self.pt]
                sentence_in = torch.tensor(sentence_idxs, dtype=torch.float).cuda()

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

            idxs = [self.pt[w] for w in doc if w in self.pt]
            inputs = torch.tensor(idxs, dtype=torch.float).cuda()
            label_scores = self(inputs)
            lix = np.argmax(label_scores[-1]).item()

            return self.ix_to_label[lix]


    def transform(self, doc):

        sentence_idxs = [self.pt[w] for w in doc if w in self.pt]
        sentence_in = torch.tensor(sentence_idxs, dtype=torch.float).cuda()
        embeds = torch.mean(sentence_in.cuda(), dim=0)

        return embeds.cpu().detach().numpy()

        # lin1_out = self.lin1(embeds.view(1, 1, -1)).cuda()
        # lin2_out = self.lin2(lin1_out.view(1, 1, -1)).cuda()
        # lin3_out = self.lin3(lin2_out.view(1, 1, -1)).cuda()
        # lin4_out = self.lin4(lin3_out.view(1, 1, -1)).cuda()
        # lin5_out = self.lin5(lin4_out.view(1, 1, -1)).cpu().detach().numpy()[0][0]

        # return lin5_out
