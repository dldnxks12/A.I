'''

    못 맞춘 것 뿐 아니라, 자신감 없이 맞춘 정답까지 페널티를 부가

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


N = 500
(X, y_org) = make_blobs(n_samples=N, n_features=2, centers=2, cluster_std=2.0, random_state=17)

x1, x2 = X[:,0], X[:,1]

y = y_org.copy()
y[y==0] = -1 # label을 -1로

w1, w2, b = np.random.randn(), np.random.randn(), np.random.randn()

for epoch in range(1000):

    w1_dev = 0
    w2_dev = 0
    b_dev  = 0
    loss   = 0

    for i in range(N):
        # y*hypothesis
        score = y[i]*(w1*x1[i] + w2*x2[i] + b)

        if score < 1: # Margin 내에 있다면
            w1_dev = w1_dev - y[i]*x1[i]
            w2_dev = w2_dev - y[i] * x2[i]
            b_dev  = b_dev - y[i]
            loss = loss + (1 - score)

        w1_dev = w1_dev / float(N)
        w2_dev = w2_dev / float(N)
        b_dev = b_dev / float(N)
        loss = loss / float(N)

        w1 = w1 - 0.01 * w1_dev
        w2 = w2 - 0.01 * w2_dev
        b  =  b - 0.01 * b_dev

    if epoch % 100 == 0:
        print(f"Loss {loss}")

ACC = (((w1*x1 + w2*x2 + b) > 0 ) == y_org).sum()/N
print(f"ACC {ACC}")

#plt.figure()
#plt.scatter(x1, x2, c = y_org)
#plt.show()


