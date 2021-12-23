import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

N = 500
(X, y) = make_blobs(n_samples=N, n_features=2, centers=2, cluster_std=2.0, random_state=17)

print(np.shape(X))
print(np.shape(y))

x1, x2 = X[:, 0], X[:, 1]

# Normal distribution에서 1개 randomly pick
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

a = 0.01
for epoch in range(10000):

    hypothesis = sigmoid(w1*x1 + w2*x2 + b)

    loss = -( ( y*np.log(hypothesis) + (1-y)*np.log(1-hypothesis) ) ).mean()

    dev = hypothesis - y
    # Gradient Descent
    w1 = w1 - (a*dev*x1).mean()
    w2 = w1 - (a*dev*x2).mean()
    b  = b  - (a*dev).mean()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss}")

prediction = ( (sigmoid(w1*x1 + w2*x2 + b) > 0.5) == y).sum() / N
print(f"ACC : {prediction*100}")

plt.figure()
plt.subplot(1,2,1)
plt.scatter(x1, x2, c = y)

plt.subplot(1,2,2)
plt.scatter(x1, x2, c = y)

xx1 = np.linspace(-10, 10, 100)
y = - (xx1*w1 + b) / w2

plt.plot(xx1, y, c = 'r')
plt.show()


