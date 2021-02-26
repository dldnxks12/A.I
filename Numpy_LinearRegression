# 다변수 함수의 Gradient Descent

import numpy as np
import matplotlib.pyplot as plt

Data = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype = np.float32)

# data slicing
# 1~3 열 입력 데이터 , 4열 정답 데이터 
# 입력 변수가 3개 이므로, Weight 또한 3개 (w1, w2, w3)

train_data = Data[:,0:3]
test_data = Data[:,[3]]
w = np.random.rand(3,1)

learning_rate = 0.000001 # 학습률이 이보다 크거나 작을 경우 Error 값이 크게 튀는 걸 확인

def loss_f(x, y): # 행렬 곱셈을 통한 Y Prediction과 Y label 값과의 차이 
    y_pred = np.dot(x,w)
    return np.sum( (y_pred - y)**2 ) / len(x)

def predict(x):  # Model이 잘 학습되었는지 실제 데이터를 넣어 확인하는 함수
    y_pred = np.dot(x,w)
    return y_pred


# 시행 횟수에 따른 Error 값 변화를 시각화하기 위한 list 선언

error =  []
trying = []

# 다변수함수의 Gradient Descent Algorithm 구현 
for epoch in range(1000):
    w[0][0] -=  learning_rate * ( np.sum( (predict(train_data) - test_data)*train_data[:,0] ) / len(train_data) )
    w[1][0] -=  learning_rate * ( np.sum( (predict(train_data) - test_data)*train_data[:,1] ) / len(train_data) )
    w[2][0] -=  learning_rate * ( np.sum( (predict(train_data) - test_data)*train_data[:,2] ) / len(train_data) )
    
    print("Epoch: ",epoch,", Error: ", loss_f(train_data, test_data), "w: ",w)
    error.append(loss_f(train_data,test_data))
    trying.append(epoch)

plt.plot(error,trying)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()
        
check = np.array([70, 73, 78])
predict(check)

