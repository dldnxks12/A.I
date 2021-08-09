### Before Machine Learning ...

Single Layer Perceptron 을 사용하여 간단한 분류 문제는 해결할 수 있다. 

    다음의 경우, 선형 영역에 대해서 분리가 가능하다. 즉, 적당한 직선 또는 평면으로 분류를 진행할 수 있다.
        
    ex) AND, NAND, OR 
    
Single Layer Perceptron 은 비선형 영역에 대해서는 문제 해결이 불가능하다.

    다음의 경우, 직선 또는 평면으로 분류를 진행할 수 없다.
    
    ex) XOR
    
![image](https://user-images.githubusercontent.com/59076451/128667610-3e40617e-36e3-447c-b20b-9fb62fc9fd7b.png)

Multi Layer Perceptron 은 비선형 영역에 대한 해답을 가지고 있다.

![image](https://user-images.githubusercontent.com/59076451/128667633-2606f7d7-6604-42ed-b18c-fdbb36d5e9a0.png)
    
---

### MLP

XOR와 같은 문제는 Multi Layer Perceptron로 해결이 가능하다. 

AND, NAND, OR 문제를 풀기 위해 사용한 함수를 섞어서 사용하면 되는데, Perceptron 관점에서는 Layer를 쌓는 것이다.

    Hypothesis = Wx + b 라는 식을 통해 항상 우리가 결과로써 직선을 얻는 것으로 알고 있다.
    
    이게 어떻게 곡선 또는 면이 될 수 있는 지 잘 생각해보자.
    
        첫 번째 Layer에 대한 좌표가 존재하고, 두 번째 Layer에 대한 좌표가 존재한다고 가정해보자.
        
        두 번째 Layer에서 출력으로 내놓은 직선을, 첫 번째 좌표에서 표현하게 되면 곡선이 된다.
        
        즉, 각 layer에서 내놓은 출력은 각각 직선의 형태지만, 해당 직선은 다른 Layer의 좌표계에서 표현하게 되면 곡선이나 면으로 표현된다. 

위의 AND, NAND, OR, XOR 문제를 해결하는 과정에서 인간이 직접 그 가중치와 편향들을 계산해서 알맞는 값을 출력하도록 해왔다.

하지만 이후로 더 복잡한 비선형 계산을 위해 MLP의 hidden layer를 늘리고, 모델이 직접 그 Weight와 Bias를 찾도록 자동화시키는 쪽으로 발전한다.

이에 Linear Regression 이나 Logistic Regression 과 같이 loss function (오차 줄이기 )과 optimizer (기울기 계산해서 업데이트하기) 를 사용한다.

이를 모델을 training 시킨다고 한다. 

---

### 활성함수로 비선형 함수를 선택해야 하는 이유 

![image](https://user-images.githubusercontent.com/59076451/128674911-40e86d52-d650-48e3-8409-0759d3ebc49d.png)


- 활성함수의 사용 위치

![image](https://user-images.githubusercontent.com/59076451/128675015-044ee335-aa0c-46af-9147-f9f8526e7e56.png)
