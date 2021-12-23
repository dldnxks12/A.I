import numpy as np
import matplotlib.pyplot as plt

n = 100
x = np.random.randn(n) # 100개의 random한 정규분포 data
y = x*20 + 10
y = y + np.random.randn(n)*10 # add noise

w = np.random.randn(1)
b = np.random.randn(1)

learning_rate = 0.01
loss_history = []

for epoch in range(1000):
    y_pred = w*x + b

    loss = ((y - y_pred)**2).mean() # Mean Squared Error

    # Gradient Update
    w = w - (learning_rate*2*(y_pred-y)*x).mean()
    b = b - (learning_rate*2*(y_pred-y)).mean()

    loss_history.append(loss)

    if epoch % 10 == 0:
        print(f"Current Epoch {epoch}, Current Loss {loss}")


plt.figure()
plt.scatter(x,y)
xx = np.linspace(-5,5,100)
yy = w*xx + b
plt.plot(xx,yy, c = 'r')

plt.show()








