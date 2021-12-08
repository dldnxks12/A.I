import os
import torch
import torch.nn as nn
from torchtext.legacy import data, datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed(777)


# Hyperparameters

batch_size = 64
learning_rate = 0.001
traininig_epochs = 5

TEXT = data.Field(sequential = True, batch_first = True, lower = True)
LABEL = data.Field(sequential = False, batch_first = True)

trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(trainset, min_freq = 5)
LABEL.build_vocab(trainset)

trainset, valset = trainset.split(split_ratio = 0.8)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (trainset, valset, testset), batch_size = batch_size, shuffle = True, repeat = False
)

vocab_size = len(TEXT.vocab)
n_classes = 2 # Positive / Negative

print("[TrainSet]: %d [ValSet]: %d [TestSet]: %d [Vocab]: %d [Classes] %d"
      % (len(trainset),len(valset), len(testset), vocab_size, n_classes))

class BasicGRU(nn.Module):
  def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p = 0.2):
    super(BasicGRU, self).__init__()
    self.n_layers = n_layers
    self.embed = nn.Embedding(n_vocab, embed_dim) # n_vocab 개수의 단어들을 embed_dim 크기의 벡터로 Embedding

    self.hidden_dim = hidden_dim
    self.dropout = nn.Dropout(dropout_p)

    self.gru = nn.GRU(embed_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True)

    self.out = nn.Linear(self.hidden_dim, n_classes)

  def forward(self, x):
    x = self.embed(x)
    x , _ = self.gru(x)

    # 해당 batch 내의 가장 마지막 hidden state
    h_t = x[:, -1, :]

    self.dropout(h_t)

    out = self.out(h_t)

    return out

model = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(traininig_epochs):
  avg_cost = 0
  for batch in train_iter:
    X, Y = batch.text.to(device), batch.label.to(device)
    Y.data.sub_(1)

    hypothesis = model(X)
    cost = criterion(hypothesis, Y)


    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    avg_cost += cost / batch_size

  print(f"Epoch : {epoch} Cost : {avg_cost}")


torch.save(model.state_dict(), '/content/drive/MyDrive/Colab Notebooks/인공지능 응용/HW/model_s1.pt')


# model load
model_new = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(device)
model_new.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/인공지능 응용/HW/model_s1.pt'))

# Validation

corrects = 0
for batch in val_iter:
  x,y = batch.text.to(device), batch.label.to(device)
  y.data.sub_(1)
  hypothesis = model_new(x)
  corrects += (hypothesis.max(1)[1].view(y.size()).data == y.data).sum()

print('accuracy = ', corrects/len(val_iter.dataset)*100.0)

# Test

corrects = 0
for batch in test_iter:
  x,y = batch.text.to(device), batch.label.to(device)
  y.data.sub_(1)
  hypothesis = model_new(x)
  corrects += (hypothesis.max(1)[1].view(y.size()).data == y.data).sum()

print('accuracy = ', corrects/len(test_iter.dataset)*100.0)

# 텍스트 입력의 숫자 변환 과정 (Text -> Digitalize)
input_text = testset[3].text
print(input_text)
for i in range(len(input_text)):
  print(TEXT.vocab[input_text[i]], end = ', ')

Temp = []
for i in range(len(input_text)):
  Temp.append(TEXT.vocab[input_text[i]])

Temp = torch.LongTensor(Temp)
Temp = Temp.unsqueeze(0).to(device)

hypothesis = model_new(Temp)
print(hypothesis.max(1)[1])