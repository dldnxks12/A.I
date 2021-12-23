import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(777)

sample_sentence = 'hi!hello.'
char_set = list(set(sample_sentence))

# 각 문자에 대해서 Index를 맞춰줄 수 있다.
dic = { c : i for i , c in enumerate(char_set)}

# Parameter # RNN에서는 Input Size, Hidden size 동일 --- One hot으로 비교할 것이기 때문
dic_size = len(dic)
input_size = dic_size
hidden_size = dic_size

# X data -> One_hot으로
x_batch = []
y_batch = []

x_data = [dic[c] for c in sample_sentence[:-1]] # 각 문자의 Index
x_one_hot = [np.eye(dic_size)[x] for x in x_data]

y_data = [dic[c] for c in sample_sentence[1:]]

x_batch.append(x_one_hot)
y_batch.append(y_data)

# Tensor Type으로
X = torch.FloatTensor(x_batch)
Y = torch.LongTensor(y_batch)

# Model
learning_rate = 0.01
training_epochs = 50
model = nn.RNN(input_size = input_size, hidden_size = hidden_size, batch_first=True)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(training_epochs):
    hypothesis, _status = model(X)
    cost = criterion(hypothesis.reshape(-1, dic_size), Y.reshape(-1))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 5 == 4:
        result = hypothesis.data.numpy().argmax(axis=2)
        result_string = ''.join([char_set[c] for c in np.squeeze(result)])

        print(f"Prediction : {result_string} | True Label : {sample_sentence[1:]}")










