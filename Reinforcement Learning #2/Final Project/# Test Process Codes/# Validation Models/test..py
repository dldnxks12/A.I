###########################################################################
# To Avoid Library Collision
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
###########################################################################

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


queue = [ 1, 2, 3, 4, 5]

b = queue.pop()
c = queue.pop()
d = queue.pop()
queue.insert(0, 6)
queue.insert(0, 8)
e = queue.pop()
f = queue.pop()
queue.pop()
print(b, c, d, e, f)
print(queue)


