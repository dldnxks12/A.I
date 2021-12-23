import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.randn(100)
x2 = np.random.randn(100)

y = x1*30 + x2*40 + 50 # W1 : 30 , W2 : 40 , Bias : 50으로 나오는 것이 정답
y = y + np.random.randn(100)*20 # Add noise to y label

w1 = np.random.randn(1)
w2 = np.random.randn(1)
b = np.random.randn(1)

learning_rate = 0.01
loss_history = []

for epoch in range(1000):

    y_pred = (w1*x1) + (w2*x2) + b

    # loss
    loss = ((y_pred - y)**2).mean()

    # Gradient update
    w1 = w1 - learning_rate* (2 * (y_pred - y) * x1).mean()
    w2 = w2 - learning_rate * (2 * (y_pred - y) * x2).mean()
    b = b - learning_rate * (2 * (y_pred - y)).mean()

    loss_history.append(loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss}")


plt.figure()
ax1 = plt.axes( projection='3d')
ax1.scatter3D(x1, x2, y)

xx = np.linspace(-3, 3, 100)
xx2 = np.linspace(-2, 2, 100)
#yy = x1*w1 + x2*w2 + b

ax1.plot(xx, xx2, xx*w1 + xx2*w2 + b , c = 'r')
plt.show()
