#### Multi class Classification

  Softmax를 이용해서 여러 개의 class도 분류할 수 있다.
  
  Sudo code
  
    1. Label Data -> One-hot Encoding
    2. nn.Linear(x_train)
    3. nn.Softmax(x_train)
    4. CrossEntropy(prediction, y_label_one_hot)
    5. update -> goto step 1

    라벨 데이터 원 핫 인코딩 - 선형 회귀 Hypothesis 생성 (Weight, Bias 학습 및 출력 Class와 차원 맞추기) - Softmax 함수 통과 (0~1 사이 확률 값으로 매핑) -  Cost 계산 - 업데이트
  
---
 
#### One-hot Encoding

  선택지의 개수만큼의 차원을 가지면서, 해당 선택지의 인덱스에 해당하는 원소는 1, 나머지는 0의 값을 가지도록 하는 표현법
  
  각 Class가 서로 균등한 관계, 즉 순서나 계층 관계가 없는 경우에 사용하기 좋다. Sequenctial 데이터와 같이 순서가 있는 경우 label을 0 1 2 3 4 식으로 주어 Weight를 갖게 하는 방법도 있다.
  
    강아지 = [1, 0, 0]
    고냥이 = [0, 1, 0]
    멍게   = [0, 0, 1]
    
  총 3개의 선택지가 있으니 3차원 Vector이다.
  
      x_train = torch.Tensor( [ [1, 0, 0]
                                [0, 1, 0]
                                [0, 0, 1] ]) 
                                
---

#### Softmax Regression

  binary classification(이진 분류) 에서는 출력 값으로 0 또는 1의 값을 갖게 했다.
   
  우리는 이제 Linear Regression으로 도출한 Hypothesis를 Sigmoid가 아니라 Softmax 라는 함수에 통과시킨다!

    입력 데이터를 Linear Regression을 통해 Class와 차원이 같은 Vector로 만들어주고, 이 Vector를 Softmax 함수에 통과시킨다.    

    Softmax regression 또는 multi class classification에서는 다양한 class에 대해서 각각 확률로써 결과값을 도출한다. 
    
    이 결과값에 대해서 Cost 를 계산해서 학습한다. (Cross Entropy)
   
![image](https://user-images.githubusercontent.com/59076451/128623574-55c2e674-52a0-4e83-b63c-3a285d03a894.png)



  
  

---

#### Softmax function

  이 함수는 분류해야 하는 클래스의 총 개수를 k라고 할 때, k 차원의 벡터를 입력으로 받고 각 클래스에 대한 확률을 추정한다. 
  
![image](https://user-images.githubusercontent.com/59076451/128623618-3bb936d9-6e0f-4b80-8925-c7b3d6c7b8f9.png)


![image](https://user-images.githubusercontent.com/59076451/128623751-756ffae3-c24d-4485-9886-0500c955e3f8.png)

---

#### Cost Function

이진 분류에서의 cost function와 비슷하다. 하지만 one-hot encoding된 vector에 대해서 loss를 계산하여야 하므로 다음과 같은 형태를 띈다.

![image](https://user-images.githubusercontent.com/59076451/128623784-a268852e-fbb6-4df2-9bf5-eef13c2d8294.png)

하지만 수식의 형태만 다르게 보일 뿐 본질적으로 이진 분류에서의 BCE와 동일하다. CE에서 class가 2개인 경우가 BCE

![image](https://user-images.githubusercontent.com/59076451/128623813-1b807551-7cc0-4500-84d8-628df551f65d.png)

