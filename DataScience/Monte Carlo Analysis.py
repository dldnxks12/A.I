import numpy as np
import matplotlib.pyplot as plt

# 단위 사각형 내에 N 개의 Point를 찍은 후 대상 Object 내에 몇 개의 Point가 찍히는 지 Check

N = 100000

x = np.random.rand(N).round(2)
y = np.random.rand(N).round(2)

# PI의 값을 찾아보는 예제

cnt = 0
for i in range(N):
    if (x[i]*x[i]) + (y[i]*y[i]) < 1:
        cnt += 1
    else:
        continue

Pi = (4 * cnt) / N
print(Pi)

plt.figure(figsize=(6,6))
plt.scatter(x, y)
a = np.linspace(0,1,100)
b = np.sqrt(1 - a*a)
plt.plot(a,b, c = 'r' )
plt.show()
