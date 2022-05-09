import sys
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 2)
Y = np.ones((100,1))

# Define 3 Class (Label : 1, 2, 3)
Y[X[:, 1] <= (3*X[:,0] - 0.5)] = 2
Y[X[:, 1] <= (3*X[:,0] - 1.5)] = 3

### Define perceptron
# Method 1 : one vs rest

# 1번 라벨 : 0 , 2번 라벨 1, 3번 라벨 2
# Method 2 : one vs one
def perceptron(X, Y):
    X_ones = np.concatenate((np.ones((len(X), 1)), X), axis=1) # 100 x 3
    W1 = np.zeros(X_ones.shape[1]) # Class 0 vs Class 1, 2
    W2 = np.zeros(X_ones.shape[1]) # Class 1 vs Class 0, 2
    W3 = np.zeros(X_ones.shape[1]) # Class 2 vs Class 0, 1

    # Class label을 잠시 1과 -1만으로 만들자
    Y_temp1 = Y.copy()
    Y_temp2 = Y.copy()
    Y_temp3 = Y.copy()

    epochs = 500
    Y_temp1[Y_temp1 == 1] = 1
    Y_temp1[Y_temp1 == 2] = -1
    Y_temp1[Y_temp1 == 3] = -1

    Y_temp2[Y_temp2 == 1] = -1
    Y_temp2[Y_temp2 == 2] = -1
    Y_temp2[Y_temp2 == 3] = 1

    Y_temp3[Y_temp3 == 1] = -1
    Y_temp3[Y_temp3 == 2] = 1
    Y_temp3[Y_temp3 == 3] = -1

    for ep in range(epochs):
        Mat_result1 = np.matmul(X_ones, W1)
        Mat_result2 = np.matmul(X_ones, W2)
        Mat_result3 = np.matmul(X_ones, W3)
        for index, value in enumerate(zip(Mat_result1,Mat_result2,Mat_result3)):
            if Y_temp1[index] * value[0] <= 0:
                W1 = W1 + (X_ones[index] * Y_temp1[index])
            if Y_temp2[index] * value[1] <= 0:
                W2 = W2 + (X_ones[index] * Y_temp2[index])
            if Y_temp3[index] * value[2] <= 0:
                W3 = W3 + (X_ones[index] * Y_temp3[index])

    return W1, W2, W3 # Return trained weights

Weight1, Weight2, Weight3 = perceptron(X, Y)
print("Weights", Weight1,Weight2,Weight3)

plt.scatter(X[:, 0], X[:, 1], c=Y, s=10, cmap=plt.cm.Paired)
plt.plot([0,1],[-Weight1[0]/Weight1[2], -Weight1[0]/Weight1[2]-Weight1[1]/Weight1[2]])
plt.plot([0,1],[-Weight2[0]/Weight2[2], -Weight2[0]/Weight2[2]-Weight2[1]/Weight2[2]])
plt.plot([0,1],[-Weight3[0]/Weight3[2], -Weight3[0]/Weight3[2]-Weight3[1]/Weight3[2]])
plt.xlim((0,1))
plt.ylim((0,1))
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()