import numpy as np
from itertools import count
import sys

test_array = np.array([1, 2, 3, 4, 5, 6])
test_array2 = np.zeros(10)

print(test_array2.shape)
print(test_array.shape)
mask = tuple(map(lambda x : x > 3, test_array))

print(mask)
print(test_array2)

test_array2[list(mask)] = test_array

print(test_array2)


