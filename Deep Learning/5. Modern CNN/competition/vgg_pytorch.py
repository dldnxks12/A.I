import os, sys
import numpy as np
from skimage import io
from network import *
from utils import *
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # for softmax
from torch.utils.data import DataLoader

def main():

    # training and evaluation device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    # TODO : saves model in this directory
    save_dir = './result'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # TODO : Name your model as prefix
    prefix = "vgg_test"
    save_model_name =  prefix + "_model.pt"
    save_model_path = os.path.join(save_dir, save_model_name)

    print("====================================")
    print("device set to  : " + str(device))
    print("model save path: " + save_model_path)
    print("====================================")

    # TODO : Make train dataset
    x_train = []
    y_train = []
    for folder_idx in range(1, 1001):
        for img_idx in range(0, 20):
            x_train.append(get_image('fake_data_1000cls/'+str(folder_idx), str(img_idx)+".png", crop = True))
            y_train.append(np.array([folder_idx]))

        # We convert data type in dataset class
        # x_train = torch.FloatTensor(x_train)
        # y_train = torch.FloatTensor(y_train)
        break

    # TODO : Define Custom Dataset and DataLoader
    dataset           = CustomDataset(x_train, y_train, device)
    train_data_loader = DataLoader(dataset=dataset, batch_size = 10, shuffle=True, drop_last=True)

    # TODO : Set your hyperparameters and define model, optimizer
    learning_rate   = 0.01
    training_epochs = 10
    model           = VGG(in_channel=3).to(device)
    optimizer       = optim.SGD(model.parameters(), lr=learning_rate)

    # TODO: train your model!
    for epoch in range(training_epochs):
        avg_cost = 0
        for x, y in train_data_loader:
            out = model(x)                 # Inference
            cost = F.cross_entropy(out, y) # Calculate cost

            # Optimizer your model based on cost
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            avg_cost += cost

        print(f" # ------- epoch {epoch+1} / {training_epochs}| avg_cost {avg_cost} ------- #")

    end_time     = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("=============== TIME ===============")
    print("started training at : " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time : " + time_elapsed)
    print("====================================")



if __name__ == '__main__':
    main()
