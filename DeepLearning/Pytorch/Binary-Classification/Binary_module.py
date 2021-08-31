'''

아래 3개의 torch API를 이용한 Binary Classification 구현 (학습 과정도 추가)

torch.sigmoid() 
torch.nn.Linear() 
torch.nn.binary_cross_entropy()  


Linear regession에서 다룬 hypothesis에 sigmoid 함수를 거쳐 나온 값이 logistic regression 에서의 가설이다.

따라서 torch.nn.Linear()를 통과해서 나온 결과 값에 sigmoid 함수를 씌워주면 logistic regression의 가설을 얻을 수 있다.

이 후 해당 값에 따라 True, False를 나누어주어야 하는 것에 주의!

'''


# nn.Sequential() 함수 사용 --- 이 함수는 nn.Module 층을 차례로 쌓을 수 있게 한다. 인공신경망에 잘 쓰이니 꼭 기억할 것  

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(1)

x_train = torch.FloatTensor( [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]] ) # 6 x 2 Tensor
y_train = torch.FloatTensor( [[0],[0],[0],[1],[1],[1]] )# 6 x 1 Tensor 

model = nn.Sequential(

    nn.Linear(2,1), # input_dim 2, output_dim 1
    nn.Sigmoid()
    
)

'''

pre_pred = model(x_train)
print(pre_pred)

tensor([[0.4020],
        [0.4147],
        [0.6556],
        [0.5948],
        [0.6788],
        [0.8061]], grad_fn=<SigmoidBackward>)


lessAPI.py에서 확인 한 초기 예측 값이랑 왜 다르지?

    - lessAPI.py에서는 Weight, bias를 0으로 초기화 시켜놓고 시작했었다.
    
    - nn.Module을 사용하면 내부적으로 weight, bias가 있고 이는 쓰레기 값으로 들어가 있기 때문에 초기 값은 의미없다

'''

# 경사하강 (기울기)
optimizer = optim.SGD(model.parameters(), lr = 1)


# training 

num_epochs = 1000
for epoch in range(num_epochs+1):
    
    # hypothesis 계산
    pred = model(x_train)
        
    # 계산된 hypothesis 와 label 의 loss
    cost = F.binary_cross_entropy(pred, y_train)
    
    optimizer.zero_grad() # 기울기 값 초기화
    cost.backward()       # 기울기 값 계산
    optimizer.step()      # 업데이트 
    
    if epoch % 10 == 0:
      
        prediction = pred >= torch.FloatTensor([0.5]) # 0.5 이상이면 True로 
        
        correct_prediction = prediction.float() == y_train # 실제 값과 일치하는 경우에 True
        
        acc = correct_prediction.sum().item() / len(correct_prediction) 
        
        print(epoch, cost.item(), acc*100)


# ---- test ---- #

new_var = torch.FloatTensor([[3,4]])
pred = model(new_var)

print(pred) # pred 는 hypothesis
