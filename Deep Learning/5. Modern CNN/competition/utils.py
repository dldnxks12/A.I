import os, sys
import numpy as np
from skimage import io
import torch
from torch.utils.data import DataLoader

# TODO : Define Dataset for DataLoader
class CustomDataset(torch.utils.data.Dataset):
    # Load data pairs
    def __init__(self, x_data, y_data, device):
        self.x_data = x_data
        self.y_data = y_data
        self.device = device

    # Return x, y items
    def __getitem__(self, idx):
        # width x height x channel -> channel x width x height
        x = torch.FloatTensor(self.x_data[idx]).transpose(0, 2).to(self.device)
        y = torch.LongTensor(self.y_data[idx]).squeeze(0).to(self.device)
        return x, y

    # Return data length
    def __len__(self):
        return len(self.x_data)

# TODO : Load Image Data from Data folders
def get_image(folder=None, file_name=None, crop = True):
    path = os.path.join(folder, file_name)
    img  = np.array(io.imread(path))

    # TODO : Image size preprocessing (example : crop, reshape, augment...)
    if crop == True:
        img = img[:90, :190, :]
    else:
        pass
    return img
