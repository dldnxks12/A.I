import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys

torch.manual_seed(777)

sample_sentence = ['howareyou', 'whats up?', 'iamgreat.']
char_set = list(set(''.join(sample_sentence))) # sample_sentence 내부의 sentence들 모두 합치기

# 각 문자에 대해서 Index를 맞춰줄 수 있다.
dic = { c : i for i , c in enumerate(char_set)}

# Parameter # RNN에서는 Input Size, Hidden size 동일 --- One hot으로 비교할 것이기 때문
dic_size = len(dic)
input_size = dic_size
hidden_size = dic_size

print(dic_size) # 17

# X data -> One_hot으로
x_batch = []
y_batch = []

for sentence in sample_sentence:
    x_data = [dic[c] for c in sentence[:-1]] # 각 문자의 Index
    x_one_hot = [np.eye(dic_size)[x] for x in x_data]

    y_data = [dic[c] for c in sentence[1:]]

    x_batch.append(x_one_hot)
    y_batch.append(y_data)


X = torch.FloatTensor(x_batch)
Y = torch.LongTensor(y_batch)

print(X.shape) # 3 x 8 x 17
print(Y.shape) # 3 x 8

# Model
learning_rate = 0.01
training_epochs = 500
model = nn.RNN(input_size = input_size, hidden_size = hidden_size, batch_first=True)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(training_epochs):
    hypothesis, _status = model(X)
    cost = criterion(hypothesis.reshape(-1, dic_size), Y.reshape(-1)) # 24 x 17 , 24

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 5 == 4:
        result = hypothesis.data.numpy().argmax(axis=2)
        for i , sentence in enumerate(result):

            result_string = ''.join([char_set[c] for c in np.squeeze(sentence)])
            print(f"Prediction : {result_string} | True Label : {sample_sentence[i][1:]}")









