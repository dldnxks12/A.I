#### Before Machine Learning ...


Single Layer Perceptron 을 사용하여 간단한 분류 문제는 해결할 수 있다. 

    다음의 경우, 선형 영역에 대해서 분리가 가능하다. 즉, 적당한 직선 또는 평면으로 분류를 진행할 수 있다.
        
    ex) AND, NAND, OR 
    
Single Layer Perceptron 은 비선형 영역에 대해서는 문제 해결이 불가능하다.

    다음의 경우, 직선 또는 평면으로 분류를 진행할 수 없다.
    
    ex) XOR
    
![image](https://user-images.githubusercontent.com/59076451/128667610-3e40617e-36e3-447c-b20b-9fb62fc9fd7b.png)

Multi Layer Perceptron 은 비선형 영역에 대한 해답을 가지고 있다.

![image](https://user-images.githubusercontent.com/59076451/128667633-2606f7d7-6604-42ed-b18c-fdbb36d5e9a0.png)
    

#### MLP

XOR와 같은 문제는 Multi Layer Perceptron로 해결이 가능하다. 

AND, NAND, OR 문제를 풀기 위해 사용한 함수를 섞어서 사용하면 되는데, Perceptron 관점에서는 Layer를 쌓는 것이다.

위의 AND, NAND, OR, XOR 문제를 해결하는 과정에서 인간이 직접 그 가중치와 편향들을 계산해서 알맞는 값을 출력하도록 해왔다.

하지만 이후로 더 복잡한 비선형 계산을 위해 MLP의 hidden layer를 늘리고, 모델이 직접 그 Weight와 Bias를 찾도록 자동화시키는 쪽으로 발전한다.

이에 Linear Regression 이나 Logistic Regression 과 같이 loss function (오차 줄이기 )과 optimizer (기울기 계산해서 업데이트하기) 를 사용한다.

이를 모델을 training 시킨다고 한다. 

