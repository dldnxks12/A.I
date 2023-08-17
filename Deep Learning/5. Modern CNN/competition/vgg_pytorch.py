import os, sys, math
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
    col_dir = './Competition/fake_data/*/*.png'

    # creating a collection with the available images
    col = io.imread_collection(col_dir)

    x_train = []
    max_h = 420
    max_w = 620
    for c in col:
        sh = np.shape(c)
        pad_h = math.ceil((max_h - sh[0])/2)
        pad_w = math.ceil((max_w - sh[1])/2)
        x_train_img = np.pad(c, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
        x_train_img = x_train_img[:420, :620, :]
        x_train_img = cv2.resize(x_train_img, (180, 180))
        x_train.append(x_train_img)

    x_train = np.array(x_train)

    y_train = []
    for folder_idx in range(10, 31):
        for img_idx in range(0, 200):
            # x_train.append(get_image('Competition/fake_data_1000cls/'+str(folder_idx), str(img_idx)+".png", crop = True))
            y_train.append(np.array([folder_idx]))

    # TODO : Define Custom Dataset and DataLoader
    dataset           = CustomDataset(x_train, y_train, device)
    train_data_loader = DataLoader(dataset=dataset, batch_size = 20, shuffle=True, drop_last=True)

    # TODO : Set your hyperparameters and define model, optimizer
    learning_rate   = 0.01
    training_epochs = 10
    model           = VGG(in_channel=3).to(device)
    optimizer       = optim.SGD(model.parameters(), lr=learning_rate)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    # TODO: train your model!
    for epoch in range(training_epochs):
        avg_cost = 0
        for x, y in train_data_loader:
            out  = model(x)                # Inference
            cost = F.cross_entropy(out, y) # Calculate cost

            # Optimize your model based on backpropagation
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
