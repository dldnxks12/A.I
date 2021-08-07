import torch

w = torch.tensor(2.0, requires_grad=True) # 해당 tensor에 대한 미분 값 (기울기) 을 저장하겠다는 의미! w.grad로 확인 가능

num_epochs = 20
for epoch in range(num_epochs+1):
  z = 2*w

  z.backward() # w에 대해 미분

  print(w.grad.item()) # w.grad에 값만 확인하려면 item() 사용 


'''
다음과 같이 미분 값이 누적되는 것을 확인할 수 있다.

따라서 미분 값을 매번 초기화 시켜주는 것이 중요 !

2.0
4.0
6.0
8.0
10.0
12.0
14.0
16.0
18.0
20.0
22.0
24.0
26.0
28.0
30.0
32.0
34.0
36.0
38.0
40.0
42.0
'''
