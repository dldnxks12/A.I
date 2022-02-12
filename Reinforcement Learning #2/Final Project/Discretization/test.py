import numpy as np
import pandas as pd
import random
import torch
import sys

ar = np.array([1,2,3,4,5])
a = random.randrange(1, 10)
b = random.randrange(0, 10)
c = random.randrange(0, 10)
print(a,b,c)
print(len(ar))

sys.exit()
random_number = random.uniform(-1, 1)
random_number2 = random.uniform(-1, 1)
random_number3 = random.uniform(-1, 1)
random_number4 = random.uniform(-1, 1)

data = {'Action 1' : random_number,
        'Action 2' : random_number2,
        'Action 3' : random_number3,
        'Action 4' : random_number4,}

array = [random_number, random_number2, random_number3, random_number4]

arr = np.arange(-1, 1, 0.05)
d = np.digitize(array, bins = arr)
#d = np.digitize(array, bins = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

action = 16*0.05
print(d)
print(action)

# 20 개 구간