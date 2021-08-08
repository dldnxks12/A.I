'''

아래 3개의 torch API를 이용한 Binary Classification 구현 (학습 과정도 추가)

torch.sigmoid() 
torch.nn.Linear() 
torch.nn.binary_cross_entropy()  


Linear regession에서 다룬 hypothesis에 sigmoid 함수를 거쳐 나온 값이 logistic regression 에서의 가설이다.

따라서 torch.nn.Linear()를 통과해서 나온 결과 값에 sigmoid 함수를 씌워주면 logistic regression의 가설을 얻을 수 있다.

이 후 해당 값에 따라 True, False를 나누어주어야 하는 것에 주의!

'''

