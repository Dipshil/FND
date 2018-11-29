
# coding: utf-8

# In[1]:


import torch
import csv
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
import os
import shutil
import string


# In[2]:


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# In[3]:


class Liar_Liar_Dataset(data.Dataset):
    
    def __init__(self, dataFile, vocab = None):
        self.data = list()
        self.label = list()
        self.one_hot = {'false':[1,0,0,0,0,0],
                       'true':[0,1,0,0,0,0],
                       'half-true':[0,0,1,0,0,0],
                       'mostly-true':[0,0,0,1,0,0],
                       'barely-true':[0,0,0,0,1,0],
                       'pants-fire':[0,0,0,0,0,1]}
        self.max_len=0
        with open(dataFile,'r') as f:
            self.full = csv.reader(f, delimiter='\t')
            for row in self.full:
                self.data.append(row[2])
                self.label.append(self.one_hot[row[1]])
            if(not vocab):
                self.word_to_ix = dict()
                self.build_vocab(self.data)
            else:
                self.word_to_ix = vocab

            self.ix_to_word = {index: w for (w, index) in self.word_to_ix.items()}
            
    def build_vocab(self,data):
        for sent in data:
            sent = sent.split()
            if(len(sent)>self.max_len):
                self.max_len=len(sent)
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
        self.word_to_ix['<unk>'] = len(self.word_to_ix)
        self.word_to_ix['<stop>'] = len(self.word_to_ix)
                    
    def prepare_sequence(self, seq, to_ix):
        idxs = [to_ix.get(w, to_ix['<unk>']) for w in seq.split()]
        return torch.tensor(idxs, dtype=torch.long)
    
    def prepare_sentence(self, seq, to_word):
        return [to_word[w.item()] for w in seq]

    def __getitem__(self, index):
        sentence = self.data[index]
        return self.prepare_sequence(sentence,self.word_to_ix), self.label[index]
    
    def __len__(self):
        return len(self.data)


# In[4]:
base_dir = '~'

trn_data = Liar_Liar_Dataset(base_dir+'/FND/data/train.tsv')
val_data = Liar_Liar_Dataset(base_dir+'/FND/data/valid.tsv', vocab = trn_data.word_to_ix)
tst_data = Liar_Liar_Dataset(base_dir+'/FND/data/test.tsv',vocab = trn_data.word_to_ix)


# In[5]:


def customBatchBuilder(samples):
    stop = trn_data.word_to_ix['<stop>']
    sentences, labels = zip(*samples)
    #seqLengths = [len(seq) for seq in sentences]
    #maxSeqLength = max(seqLengths)
    #sorted_list = sorted(zip(sentences, labels, seqLengths), key = lambda x: -x[2])
    #sentences, labels, seqLengths = zip(*sorted_list)
    labels = torch.tensor(list(labels),dtype = torch.float)
    
    paddedSeqs = torch.LongTensor(len(sentences), trn_data.max_len)
    paddedSeqs.fill_(stop)
    for (i, seq) in enumerate(sentences):
        paddedSeqs[i, :len(seq)] = seq
    return paddedSeqs.t().to(device), labels.to(device)

batch = 8
trainLoader = data.DataLoader(trn_data, batch_size = batch, 
                              shuffle = True, num_workers = 0,
                              collate_fn = customBatchBuilder)
valLoader = data.DataLoader(val_data, batch_size = batch, 
                              shuffle = True, num_workers = 0,
                              collate_fn = customBatchBuilder)

index, (paddedSeqs, labels) = next(enumerate(trainLoader))
print(paddedSeqs.shape)
print(labels.shape)


# In[6]:


class Liar_CNN(nn.Module):

    def __init__(self, embedding_dim, vocab_size):
        super(Liar_CNN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv_2 = nn.Conv2d(1,128,2)
        self.conv_3 = nn.Conv2d(128,128,3)
        self.conv_4 = nn.Conv2d(128,128,4)
        
        self.pool = nn.MaxPool2d(3)
        self.dropout = nn.Dropout(0.8)
        self.activation = nn.Softmax(dim=1)
        self.Linear = nn.Linear(128*153*98, 6).to(device)

    def forward(self, sentence):
        self.batch_size = sentence.size(1)
        embeds = self.word_embeddings(sentence)
        out = self.conv_2(embeds.view(self.batch_size,1,sentence.size(0),-1))
        
        out = self.conv_3(out)
        out = self.conv_4(out)
        
        out = self.pool(out)
        #print(out.shape)
        out = self.Linear(out.view(self.batch_size,-1))
        out = self.dropout(out)
        pred = self.activation(out)
        
        return pred


# In[7]:


model = Liar_CNN(embedding_dim=300, vocab_size=len(trn_data.word_to_ix)).to(device)

index, (sentence,label) = next(enumerate(trainLoader))


# In[8]:


def train(trainLoader,valLoader, n_epochs=1):
    
    model = Liar_CNN(embedding_dim=300, vocab_size=len(trn_data.word_to_ix)).to(device)
    loss_function = nn.MSELoss().to(device)
    train_loss = []
    val_loss = []
    
    optimizer = optim.SGD(model.parameters(), lr=0.005)
        
    for epoch in range(n_epochs):  
        model.train()
        cum_loss = 0.0
        t = tqdm(trainLoader, desc = 'Training epoch %d' % (epoch+1))
        for (i,(sentence,target)) in enumerate(t):
            model.zero_grad()
  
            pred = model(sentence)
            loss = loss_function(pred, target)
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            t.set_postfix(cum_loss = cum_loss / (1 + i))
        train_loss.append(cum_loss/(1+i))
        cum_loss = 0.0
        t = tqdm(valLoader, desc = 'Validation epoch %d' % (epoch+1))
        model.eval()
        for (i,(sentence,target)) in enumerate(t):
            with torch.no_grad():
                pred = model(sentence)
                loss = loss_function(pred, target)

                cum_loss += loss.item()
                t.set_postfix(cum_loss = cum_loss / (1 + i))
        val_loss.append(cum_loss/(1+i))
    path = base_dir+'/FND/results/model.pkl'
    torch.save(model.state_dict(),path)
    return model,train_loss,val_loss 


# In[9]:

trained_model,train_loss,val_loss = train(trainLoader,valLoader,n_epochs=10)


# In[ ]:


trained_model = Liar_CNN(embedding_dim=300, vocab_size=len(trn_data.word_to_ix)).to(device)
trained_model.load_state_dict(torch.load('~/FND/results/model.pkl'))
tstLoader = data.DataLoader(tst_data, batch_size = 1, 
                              shuffle = True, num_workers = 0,
                              collate_fn = customBatchBuilder)
t = tqdm(tstLoader, desc = 'Testing Model')
loss_function = nn.MSELoss().to(device)
trained_model.eval()
cum_loss=0.0
for (i,(sentence,target)) in enumerate(t):
    if(len(sentence)<9):
        continue
    with torch.no_grad():
        pred = trained_model(sentence)
        loss = loss_function(pred, target)

        cum_loss += loss.item()
        t.set_postfix(cum_loss = cum_loss / (1 + i))
print(cum_loss/(1+i))

