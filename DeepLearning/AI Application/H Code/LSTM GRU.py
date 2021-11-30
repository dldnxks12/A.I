import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(777)

sample_sentence_1 = "if you want to build a ship, don't drum up people together to "
sample_sentence_2 = "collect wood and don't assign them tasks and work, but rather "
sample_sentence_3 = "teach them to long for the endless immensity of the sea."

sample_sentence = sample_sentence_1 + sample_sentence_2 + sample_sentence_3
char_set = list(set(sample_sentence)) # Character Set
dic = { c : i for i, c in enumerate(char_set)}

dic_size = len(dic) # 25
input_size = dic_size
hidden_size = dic_size*2
output_size = dic_size

unit_sequence_length = 20

# Unit_sequence_length 만큼의 Window 크기로 이동하며 Sequence를 잘라서 여러 개 batch로 이루어진 X, Y를 만든다.
input_batch = []
output_batch = []

batch_sequence = []

for i, _ in enumerate(sample_sentence[:-20]):
  batch_sequence.append(sample_sentence[i:i+21])

for batch in batch_sequence:
  x_data = [dic[c] for c in batch[:-1]]
  x_one_hot = [np.eye(dic_size)[x] for x in x_data]
  y_data = [dic[c] for c in batch[1:]]

  input_batch.append(x_one_hot)
  output_batch.append(y_data)
  
X = torch.FloatTensor(np.array(input_batch))
Y = torch.LongTensor(np.array(output_batch))  

# Model
class Custom_RNN(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, layers):
    super(Custom_RNN, self).__init__()
    self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers)
    self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)

  def forward(self, x):
    x, _status = self.rnn(x)
    x = self.fc(x)
    return x

class Custom_LSTM(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, layers):
    super(Custom_LSTM, self).__init__()
    self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers)
    self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)

  def forward(self, x):
    x, _status = self.rnn(x)
    x = self.fc(x)
    return x

class Custom_GRU(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, layers):
    super(Custom_GRU, self).__init__()
    self.rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers=layers)
    self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)

  def forward(self, x):
    x, _status = self.rnn(x)
    x = self.fc(x)
    return x

learning_rate = 0.05
training_epochs = 150
model_RNN = Custom_RNN(input_size, hidden_size, output_size, 2)
model_LSTM = Custom_LSTM(input_size, hidden_size, output_size, 2)
model_GRU = Custom_GRU(input_size, hidden_size, output_size, 2)

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model_RNN.parameters(), lr = learning_rate)
optimizer2 = optim.Adam(model_LSTM.parameters(), lr = learning_rate)
optimizer3 = optim.Adam(model_GRU.parameters(), lr = learning_rate)

for epoch in range(training_epochs):
optimizer1.zero_grad()
optimizer2.zero_grad()
optimizer3.zero_grad()

outputs1 = model_RNN(X)
outputs2 = model_LSTM(X)
outputs3 = model_GRU(X)

loss1 = criterion(outputs1.reshape(-1, dic_size), Y.reshape(-1))
loss2 = criterion(outputs2.reshape(-1, dic_size), Y.reshape(-1))
loss3 = criterion(outputs3.reshape(-1, dic_size), Y.reshape(-1))

loss1.backward()
loss2.backward()
loss3.backward()

optimizer1.step()
optimizer2.step()
optimizer3.step()

if epoch % 10 == 9:
  print(f"Epoch : {epoch} | Loss RNN : {loss1.item()} | Loss LSTM : {loss2.item()} | Loss GRU : {loss3.item()} ")
    
    
results1 = outputs1.data.numpy().argmax(axis = 2)
results2 = outputs2.data.numpy().argmax(axis = 2)
results3 = outputs3.data.numpy().argmax(axis = 2)    

result_sentence1 = []
for i,  character in enumerate(results1):
  print(char_set[character[0]], end = '')
  result_sentence1.append(char_set[character[0]])
  if i == len(results1) - 1:
      for sen in character:
        print( char_set[sen], end = '')
        result_sentence1.append(char_set[sen])

result_sentence2 = []
for i,  character in enumerate(results2):
  print(char_set[character[0]], end = '')
  result_sentence2.append(char_set[character[0]])
  if i == len(results2) - 1:
      for sen in character:
        print( char_set[sen], end = '')
        result_sentence2.append(char_set[sen])
        
result_sentence3 = []
for i,  character in enumerate(results3):
  print(char_set[character[0]], end = '')
  result_sentence3.append(char_set[character[0]])
  if i == len(results3) - 1:
      for sen in character:
        print( char_set[sen], end = '')
        result_sentence3.append(char_set[sen])        
        
result_sentence1 = ''.join(result_sentence1)
result_sentence2 = ''.join(result_sentence2)
result_sentence3 = ''.join(result_sentence3)        

from difflib import SequenceMatcher

ratio1 = SequenceMatcher(None, result_sentence1, sample_sentence).ratio()
ratio2 = SequenceMatcher(None, result_sentence2, sample_sentence).ratio()
ratio3 = SequenceMatcher(None, result_sentence3, sample_sentence).ratio()

print(ratio1, ratio2, ratio3)
