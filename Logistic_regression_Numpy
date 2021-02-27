import numpy as np

Data = np.loadtxt('diabetes.csv', delimiter=',')

x_data = Data[:,0:-1]
y_data = Data[:,[-1]]
w = np.random.rand(8,1)
learning_rate = 0.1

def sigmoid(x): # Sigmoid 
    return 1/(1 + np.exp(-x))

def loss_f(x, y): # Loss 함수는 Linear regression에서 sigmoid 함수를 한 번 더 씌워준 것
    delta = 1e-3
    z=np.dot(x,w)
    y_pred=sigmoid(z)
    return -(np.sum(y*np.log(y_pred+delta)+(1-y)*np.log((1-y_pred)+delta))) / len(x) # Cross Entropy 함수

def predict(x):
    y_pred = sigmoid(np.dot(x,w))
    
    result = 0
    
    if y_pred > 0.5 :
        result = 1
    else:
        result = 0
        
    return result

f = lambda x : loss_f(x_data, y_data)  # 람다 표현식으로 함수 생성 -> 미분에 사용할 것

def Numerical_derivative(f , w): # 수치적 미분 사용  -> f(x+h)-f(x-h) / 2h 

    delta = 1e-4
    gradient = np.zeros_like(w)  # weight array와 같은 크기를 가진 빈 값의 Gradient 생성
    
    for idx in range(w.size): #  8개의  weight들을 모두 업데이트 
        tmp = w[idx] # 해당 Index의 Weight의 변화값만을 구하고 다시 원상복구 시킬 것 이므로 temp 변수에 해당 Weight 임시 저장
        
        w[idx] = float(tmp) + delta
        fx1 = f(w)  # 해당하는 idx의 Weight를 조금 변화시킨 후의 loss ( + delta)  

        w[idx] = float(tmp) - delta
        fx2 = f(w)  # 해당하는 idx의 Weight를 조금 변화시킨 후의 loss ( - delta)
        
        w[idx] = tmp # 해당 index 값 원상 복귀
        
        gradient[idx] = (fx1 - fx2) / 2*delta # 해당 idx를 제외한 weight 값들은 모두 동일 하므로, 변화시킨 idx의 weight 값에 대한 차이만 Gradient[idx]에 저장
        
    return gradient

def accuracy(x, y): # 추측값과 정답 값이 같다면 +1 
    ACC = 0
    for x_,y_ in zip(x,y):
        if predict(x_) == y_:
            ACC += 1
    result = ( ACC/len(x) ) * 100
    return result
            
    
for epoch in range(1001):
    w -= learning_rate*Nuerical_derivative(f,w) # Weight Update 
    print("Epoch: ", epoch, "/1000" , "Error: ", loss_f(x_data,y_data))

print("Accuracy : ", accuracy(x_data,y_data), "%")
