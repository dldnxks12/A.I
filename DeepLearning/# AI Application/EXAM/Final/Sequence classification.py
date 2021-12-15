import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy import data, datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)