#### Logistic Regression : 이진 분류 

    선형 회귀에서 사용한 가설에 추가적으로 sigmoid() 활성함수를 거쳐 출력을 낸다. 
    
    다시말해 Linear regession에서 다룬 hypothesis에 sigmoid 함수를 거쳐 나온 값이 logistic regression 에서의 가설이다.

    따라서 torch.nn.Linear()를 통과해서 나온 결과 값에 sigmoid 함수를 씌워주면 logistic regression의 가설을 얻을 수 있다.

    이 후 해당 값에 따라 True, False를 나누어주어야 하는 것에 주의!
    
      { (linear regression)forward() -> sigmoid() }-> backward() 
      
#### Cost function

  1. Linear Regression : MSE
  2. Logistic Regression : BinaryCrossEntropy


    
    
    project2에서 결과로 나오는 class가 2개이기 때문에 logistic regression 사용할 것 
