## Machine Learning 

---

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
    
---

#### MLP

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

#### 활성함수로 비선형 함수를 선택해야 하는 이유 

![image](https://user-images.githubusercontent.com/59076451/128674911-40e86d52-d650-48e3-8409-0759d3ebc49d.png)


![image](https://user-images.githubusercontent.com/59076451/128675331-77c95643-6856-4bab-be4f-60f622318042.png)


- 활성함수의 사용 위치

![image](https://user-images.githubusercontent.com/59076451/128675015-044ee335-aa0c-46af-9147-f9f8526e7e56.png)


#### Overfitting Solver

[Overfitting](https://wikidocs.net/60751)

데이터 양 늘리기   - Data Augmentation  

모델 복잡도 줄이기 - Architecture Simplify 

드롭 아웃         - Drop out

        학습 과정에서 신경망의 일부를 사용하지 않는 방법
        신경만 학습 시에만 사용하고, 예측 시에는 사용하지 앟는 것이 일반적이다.
        
        장점 및 사용 이유 : 학습 시에 인공신경망이 특정 뉴런 또는 조합에 너무 의존적이게 되는 것을 방지해준다.
        
                            매번 랜덤으로 뉴런들을 선택해서 사용하므로 서로 다른 신경망을 앙상블하는 것과 같은 효과를 준다.

가중치 규제       - Regularization (Normalization과 혼동 X)
    
        복잡한 모델이 간단한 모델보다 Overfitting될 가능성이 높다. (간단한 모델 매개변수 수 < 복잡한 모델 매개변수 수)
        복잡한 모델을 간단하게 만드는 방법 중 하나로 가중치 규제가 있다.
 
        * L1 규제 : 가중치 Weight의 절대값 합계를 Cost function에 추가     λ|weight|          - L1 Norm         
        
        * L2 규제 : 가중치 Weight의 절대갑 합계의 제곱을 Cost function에 추가 1/2*λ(weight)^2  - L2 Norm
        
        λ는 규제의 강도를 정하는 변수이다. 
        λ가 크다면 모델이 훈련 데이터에 대해 적합한 Weight, Bias를 찾는 것 보다 규제를 위해 추가된 항들을 작게 유지하는 것을 우선한다는 의미이다. 
        
        각 각의 값은 기존 Cost function에 더해서 발전된 Cost function의 형태를 갖는다.
        위 두 식 모두 Cost function을 최소화 하기 위해서 가중치 Weight들이 작아져야 한다는 특징이 있다.
        
        L1 규제를 예로 들면, 비용 함수가 최소가 되게 하는 Weight와 Bias를 찾는 동시에, 가중치들의 절대값 합도 최소가 되는 방향으로 학습된다.
        다시말해서..        
             1. 알맞은 Weight, Bias들은 오차가 최소가 되도록 한 후 그대로 놔두는 방향으로 학습된다.
             2. 알맞지 않은 Weight, Bias들은 값이 거의 0에 가깝도록 조정되어 모델에 거의 사용되지 않게 된다.

             ex) H(x) = w1x1 + w2x2 + w3x3 라는 수식이 있을 때, 여기에 L1 규제를 사용했더니 w3 값이 거의 0이 되었다고 한다.
             이는 x3 라는 데이터가 사실 해당 모델의 결과에 별 영향을 주지 못하는 특성임을 의미한다.
             
        L2 규제는 Weight들의 제곱합을 최소화하므로, Weight값이 완전 0이 되기보다는 0에 가까워지는 경향을 띈다.              
        
        여기서 L1과 L2의 사용처가 조금 달라진다.
        
        L1 : 어떤 특성들이 모델에 영향을 주고 있는지? 를 정확히 판단하고자 할 때 유용하다
        L2 : 이런 판단이 필요없다면 L2 규제가 더 잘 동작하므로, L2 규제를 권장한다. 
        
            Deep Learning에서는 L2 규제를 Weight decay라고도 부른다. 
            
![image](https://user-images.githubusercontent.com/59076451/128677581-499e90d5-7876-456e-9c07-a8f8240b37a7.png)
            
#### Gradient 

[Gradient](https://wikidocs.net/61271)
   
        
