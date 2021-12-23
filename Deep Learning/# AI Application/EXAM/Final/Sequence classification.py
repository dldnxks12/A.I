import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy import data, datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

batch_size = 64
learning_rate = 0.001
training_epochs = 5

TEXT  = data.Field(sequential=True, batch_first = True, lower = True)
LABEL = data.Field(sequential=False, batch_first = True)

trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(trainset, min_freq = 5)
LABEL.build_vocab(trainset) # LABEL data 기반으로 Vocab 생성

trainset, valset = trainset.split(split_ratio = 0.9)
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (trainset, valset, testset), batch_size = batch_size, shuffle = True, repeat = False
)

vocab_size = len(TEXT.vocab)
n_classes = 2

class BasicGRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p = 0.2):
        super(BasicGRU, self).__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab, embed_dim)

        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)

        self.gru = nn.GRU(embed_dim, self.hidden_dim, num_layers = self.n_layers, batch_first= True)

        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):

        x = self.embed(x)
        x, _ = self.gru(x)

        # 가장 마지막 Hidden layer의 문장
        h_t = x[:, -1, :]

        self.dropout(h_t)

        out = self.out(h_t)

        return out

model = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(training_epochs):
    avg_cost = 0
    for batch in train_iter:
        X, Y = batch.text.to(device), batch.label.to(device)
        Y.data.sub(-1)

        hypothesis = model(X)
        cost = criterion(hypothesis, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost.item()/batch_size






