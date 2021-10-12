from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

n=100
x1 = np.random.randn(n)             # randn=normal distribution in (-1,1), rand=(0,1)
x2 = np.random.randn(n)

X = np.concatenate([x1.reshape(n,1), x2.reshape(n,1)], axis = 1) # 각 element에 list 하나 씌워주는 과정 : [1,2,3] -> [[1],[2],[3]]
y = x1*30 + x2*40 + 50
y = y + np.random.randn(n)*20

model = LinearRegression()
model.fit(X, y)

print("Score", model.score(X, y))
print(f"w1 = {model.coef_[0]}, w2 = {model.coef_[1]} , b = {model.intercept_}")

new_x = [1,3]
print(model.predict([new_x]))

w1 = model.coef_[0]
w2 = model.coef_[0]
b = model.intercept_

plt.figure()
ax1 = plt.subplot(1,2,1, projection='3d')
ax1.scatter(x1, x2, y)

xx1 = np.linspace(-3,3,100)
xx2 = np.linspace(-2,2,100)

ax1.plot(xx1,xx2, xx1*w1 + xx2*w2 + b , c = 'r' )
plt.show()


