import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(777)

sample_sentences = ['howareyou', 'whats up?', 'iamgreat.']
sample_result = ''.join(sample_sentences)
char_set = list(set(sample_result))

dic = {c:i for i , c in enumerate(char_set)}

dic_size = len(dic)
embedding_size = dic_size
hidden_size = dic_size

input_batch = []
target_batch = []

for sentence in sample_sentences:

    x_data = [dic[c] for c in sentence[:-1]]
    x_onehot = [np.eye(dic_size)[x] for x in x_data]
    y_data = [dic[c] for c in sentence[1:]]

    input_batch.append(x_onehot)
    target_batch.append(y_data)

X = torch.FloatTensor(input_batch)
Y = torch.LongTensor(target_batch)

print(X.shape) # 3 x 8 x 17
print(Y.shape) # 3 x 8

learning_rate = 0.05
training_epochs = 500

model = nn.RNN(embedding_size, hidden_size, batch_first = True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(training_epochs):

    outputs, _status = model(X)

    loss = criterion(outputs.reshape(-1, dic_size), Y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if epoch % 10 == 0:
        print(f"Epoch : {epoch} | loss : {loss.item()}")

    if epoch % 100 == 0:
        print("# ------ Check ----- # ")
        result = torch.argmax(outputs, axis = 2)
        for sentence in result:
            print(''.join([char_set[c] for c in sentence.squeeze(0)]))
        print("# ------ Check ----- # ")

