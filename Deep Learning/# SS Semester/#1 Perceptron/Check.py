import sys
import numpy as np
import matplotlib.pyplot as plt

plt.plot([0,1], [0, 20]) # [Xmin, Xmax] , [Ymin, Ymax]
plt.xlim((0,1))
plt.ylim((0,100))
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
