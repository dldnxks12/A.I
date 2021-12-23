# 과거의 입력이 현재의 입력에도 영향을 미치는 구조

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(777)

# Dataset Construction --- Text Data Embedding
sample_sentence = 'hi!hello.'
char_set = list(set(sample_sentence))

dic = { c : i for i , c in enumerate(char_set)}

dic_size = len(dic)
embedding_size = dic_size # one-hot encoding 되는 size
hidden_size = dic_size    # Output Size로 Softmax를 통해 char_set과 비교해서 tranining할 것

x_batch = []
y_batch = []

x_data = [dic[c] for c in sample_sentence[:-1]]
y_data = [dic[c] for c in sample_sentence[1:]]

x_onehot = [np.eye(dic_size)[x] for x in x_data]

x_batch.append(x_onehot)
y_batch.append(y_data)

X = torch.FloatTensor(x_batch)
Y = torch.LongTensor(y_batch)

print(X.shape)
print(Y.shape)

# Create Model
learning_rate = 0.1
traininig_epochs = 50
model = nn.RNN(embedding_size, hidden_size, batch_first = True)

# Training

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(traininig_epochs):

    outputs, _status = model(X) # _status = final hidden state result of batch
    loss = criterion(outputs.reshape(-1, dic_size), Y.reshape(-1)) # 8 x 7  | 8

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 4:
        # method 1
        result = outputs.data.numpy().argmax(axis = 2) # outputs : 1 x 8 x 7 ---
        result_str = ''.join([char_set[c] for c in np.squeeze(result)])  # np.squeeze(result) = 8 x 7

        # method 2
        #result = torch.argmax(outputs, axis = 2)
        #result_str = ''.join([char_set[c] for c in result.squeeze(0)])  # np.squeeze(result) = 8 x 7

        print(f" Loss : {loss.item()} | Prediction : {result_str} | True Y : {sample_sentence[1:]}")
