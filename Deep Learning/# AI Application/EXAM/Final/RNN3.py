import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(777)

sample_sentence = 'hi!hello.'
char_set = list(set(sample_sentence))

dic = {c : i for i, c in enumerate(char_set)}

dic_size = len(dic)
input_size = dic_size
hidden_size = dic_size

x_batch = []
y_batch = []

x_data = [dic[c] for c in sample_sentence[:-1]]
x_ont_hot = [np.eye(dic_size)[x] for x in x_data]
y_data = [dic[c] for c in sample_sentence[1:]]

x_batch.append(x_ont_hot)
y_batch.append(y_data)

X = torch.FloatTensor(x_batch)
Y = torch.LongTensor(y_batch)

learning_rate = 0.01
training_epochs = 50
model = nn.RNN(input_size = input_size, hidden_size = hidden_size, batch_first = True).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.paramters(), lr = learning_rate)

for epoch in range(training_epochs):

    X = X.to(device)
    Y = Y.to(device)

    hypothesis , _status = model(X)
    cost = criterion(X.reshape(-1, dic_size), Y.reshape(-1))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 5 ==4:
        result = hypothesis.data.numpy().argmax(axis = 2)
        result_string = ''.joint([char_set[c] for c in np.squeeze(result)])

