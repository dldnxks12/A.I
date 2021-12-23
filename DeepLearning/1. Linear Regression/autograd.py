import torch

# 값이 2인 스칼라 텐서 w 선언
# requires_grad = True는 이 tensor에 대한 기울기를 저장하겠다는 의미 -- w.grad에 w 대한 미분 값이 저장된다. 

w = torch.tensor(2.0, requires_grad=True)

y = w**2
z = 2*y + 5 # z = 2*w^2 + 5

# 해당 수식을 w에 대해서 미분해보자
# .backward() 를 호출하면 해당 수식의 w에 대한 기울기를 계산한다

z.backward()

# 이제 w.grad()를 출력하면 w가 속한 수식을 w로 미분한 값이 저장된 것을 확인할 수 있다 !!

print(w.grad.item()) # 8.0
