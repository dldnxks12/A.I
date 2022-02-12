import numpy as np
import pandas as pd
import random
import torch
import sys
import math

arr = np.array([1])
arr2 = np.array([10])
arr3 = np.array([100])
arr_ = np.array([[arr, arr2, arr3]])

print(arr_)
print(arr_.shape)

a = math.exp(-0.1065)
b = math.exp(-0.1059)
c = math.exp(0.0893)

arr4 = np.array([a])
arr5 = np.array([b])
arr6 = np.array([c])
arr__ = np.array([[arr4, arr5, arr6]])

print(arr__)
print(arr__.shape)

result = arr_ * arr__
print(result)
print(result.shape)
sys.exit()

result = c / (a+b+c)

print(result)
'''
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

'''