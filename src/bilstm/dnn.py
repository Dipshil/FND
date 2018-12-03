import torch.nn.functional as F
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch


class DNN(nn.Module):


    def __init__(self, embedding_dim, hidden_dim, lr):

        super(DNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lr = lr

        self.word_to_ix = {}
        self.label_to_ix = {}


    def forward(self, sentence):

        embeds = torch.mean(self.word_embeddings(sentence).cuda(), dim=0)
        lin1_out = self.lin1(embeds.view(1, 1, -1)).cuda()
        lin2_out = self.lin2(lin1_out.view(1, 1, -1)).cuda()
        lin3_out = self.lin3(lin2_out.view(1, 1, -1)).cuda()
        lin4_out = self.lin4(lin3_out.view(1, 1, -1)).cuda()
        lin5_out = self.lin5(lin4_out.view(1, 1, -1)).cuda()
        label_space = self.hidden2tag(lin5_out.view(1, -1)).cuda()
        label_scores = F.log_softmax(label_space, dim=1).cuda()

        return label_scores


    def init(self, docs):

        self.init_vocab(docs)
        self.word_embeddings = nn.Embedding(len(self.word_to_ix), self.embedding_dim).cuda()
        torch.nn.init.xavier_uniform_(self.word_embeddings.weight)
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


    def transform(self, doc):

        sentence_idxs = [self.word_to_ix[w] for w in doc if w in self.word_to_ix]
        sentence_in = torch.tensor(sentence_idxs, dtype=torch.long).cuda()
        embeds = torch.mean(self.word_embeddings(sentence_in).cuda(), dim=0)
        return embeds.cpu().detach().numpy()
        # lin1_out = self.lin1(embeds.view(1, 1, -1)).cuda()
        # lin2_out = self.lin2(lin1_out.view(1, 1, -1)).cuda()
        # lin3_out = self.lin3(lin2_out.view(1, 1, -1)).cuda()
        # lin4_out = self.lin4(lin3_out.view(1, 1, -1)).cuda()
        # lin5_out = self.lin5(lin4_out.view(1, 1, -1)).cpu().detach().numpy()[0][0]

        # return lin5_out
