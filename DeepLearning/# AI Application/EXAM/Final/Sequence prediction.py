import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# for reproducibility
torch.manual_seed(100)

# Dictionary
sample_sentence_1 = "if you want to build a ship, don't drum up people together to "
sample_sentence_2 = "collect wood and don't assign them tasks and work, but rather "
sample_sentence_3 = "teach them to long for the endless immensity of the sea."
sample_sentence = sample_sentence_1 + sample_sentence_2 + sample_sentence_3
char_set = list(set(sample_sentence))
dic = {c: i for i, c in enumerate(char_set)}

# Parameters
dic_size = len(dic)
input_size = dic_size
hidden_size = dic_size * 2
output_size = dic_size
unit_sequence_length = 20

# Dataset setting
input_batch = []
target_batch = []

make_batch = []

for i in range(len(sample_sentence)):
    if i < len(sample_sentence) - 18:
        make_batch.append(sample_sentence[i : i+19])
    else:
        break

for sentence in make_batch:
    x_data = [dic[c] for c in sentence[:-1]]
    x_one_hot = [np.eye(dic_size)[x] for x in x_data]
    y_data = [dic[c] for c in sentence[1:]]

    input_batch.append(x_one_hot)
    target_batch.append(y_data)

X = torch.FloatTensor(input_batch)
Y = torch.LongTensor(target_batch)

print(X.shape)
print(Y.shape)

class Custom_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Custom_model, self).__init__()
        self.LSTM = nn.LSTM(input_dim, hidden_dim, num_layers=layers)
        self.fc = nn.Linear(hidden_dim, output_dim, bias = True)

    def forward(self, x):
        out, status_ = self.LSTM(x)
        out = self.fc(out)

        return out


model = Custom_model(input_size, hidden_size, output_size, 2)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
criterion = nn.CrossEntropyLoss()

for epoch in range(500):
    hypothesis = model(X)
    cost = criterion(hypothesis.reshape(-1, dic_size), Y.reshape(-1))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f" Cost : {cost.item()}")


# Check Result

result = hypothesis.data.numpy().argmax(axis = 2)

result_sentence1 = []
for i,  character in enumerate(result):
  print(char_set[character[0]], end = '')
  result_sentence1.append(char_set[character[0]])
  if i == len(result) - 1:
      for sen in character:
        print( char_set[sen], end = '')
        result_sentence1.append(char_set[sen])






