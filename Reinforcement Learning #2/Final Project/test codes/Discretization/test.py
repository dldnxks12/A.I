import numpy as np
#import pandas as pd
import random
import torch
import sys
import math

a = np.random.choice([0,1,2], 1)

print(a)

b = torch.zeros((3, 1))
c = torch.zeros((1,1)) + 10

print(b.shape)
print(c.shape)
print(b)
print(c)
x = torch.cat([b, c], dim = 0)
print(x)