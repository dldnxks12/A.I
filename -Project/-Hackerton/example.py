import numpy as np


arr = np.ones((100,60))

li = []
for i in range(10):
    arr3 = arr[:, i:i+6]
    li.append(arr3)

result = np.concatenate(li, axis = 0)

print(result.shape)



