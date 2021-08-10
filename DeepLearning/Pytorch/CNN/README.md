### Convolutional Neural Network 

- 핵심 개념, flow 그리고 사용법 이해 

---

#### Convolution and Pooling

CNN은 크게 Convolution layer와 Pooling layer로 구성된다. 

Convolution layer

    Convolution 연산 -> 활성 함수 통과 (주로 ReLU)
    
Pooling layer

    Pooling 연산 

---

#### Why CNN? Why not MLP?

CNN은 이미지 처리에 특화된 신경망이다.

이미지를 기존 MLP로 처리한다고 하면, 이미지를 1차원 vector(Tensor)로 바꾸어 입력을 넣어주어야 한다. 

    1차원으로 변환된 결과는 사람이 보기에도 이게 원래 이미지였는지 알아보기 어렵다. 

![image](https://user-images.githubusercontent.com/59076451/128822916-15d5e336-e8fc-4b9c-8d08-3b4603452e0e.png)

**이와 같이 결과는 변환 전에 가지고 있던 공간적인 구조 정보가 유실된다.**

    공간적인 구조 정보 
    
    : 거리가 가까운 어떤 픽셀들끼리는 어떤 연관이 있고, 어떤 픽셀들끼리는 값이 비슷하고.. 등의 정보를 포함한 정보
    
결국 이 공간적인 구조 정보를 유지한 채 학습을 진행하기 위한 방법으로 CNN(합성곱 신경망)을 사용하게 된다.  

---

#### Channel (or Depth)

이미지는 (높이 x 넓이 x 채널)의 3차원 Tensor이다. 

Image      = Height x Width x Channel 

GrayImage  = Height x Width x 1 channel

ColorImage = Height x Width x 3 channel (R G B)

---

#### Convolution 연산

- 2차원 Tensor의 경우 (1 channel)

Convolution layer에서는 Convolution 연산을 통해 **이미지의 특징을 추출** 하는 역할을 한다. 

Kernel 또는 Filter 라는 n x m 크기의 행렬로 Height x Width 크기의 이미지를 처음부터 끝까지 겹치며 훑는다. 

    n x m 크기의 겹쳐지는 부분의 각 이미지와 커널의 원소의 값을 곲해서 모두 더한 값을 출력으로 한다. 
      
![image](https://user-images.githubusercontent.com/59076451/128824283-0495eaad-d7ae-4f48-a42a-c6994143df97.png)

- 3차원 Tensor의 경우 (3 channel 이상)

실제로 Convolution 연산의 입력은 다수의 채널을 가진 이미지 또는 이전 연산의 결과로 나온 특성 맵일 수 있다.

**다수의 채널을 가진 입력 데이터에 대해 Convolution 연산을 한다고 하면 Kernel의 channel 수도 더이상 1개가 아니라 입력 채널 수 만큼 존재해야한다.**

    입력 데이터의 Channel = Kernel의 Channel
    
Channel 수가 같으므로 Convolution 연산을 Channel 마다 수행한다. 이 후 이 결과를 모두 더해서 최종 특성 맵을 얻는다.     
    
![image](https://user-images.githubusercontent.com/59076451/128826047-1f39981e-1f39-4072-99ba-8ccd0d897569.png)

위 그림은 3개의 Channel을 가진 입력 데이터와 3개의 Channel을 가진 1개의 Kernel의 합성곱 연산을 보여준다.

주의할 점은 위에서 사용한 Kernel은 3개의 Kernel이 아니라, 3개의 Channel을 가진 1개의 Kernel이라는 점이다.
    
![image](https://user-images.githubusercontent.com/59076451/128827083-b32473e3-9369-44dd-bcb2-03dd67a493e7.png)

![image](https://user-images.githubusercontent.com/59076451/128827116-f5fd50ab-959d-49ab-b0ca-110afe9784d1.png)

---
    
#### Feature map, Stride, Padding 

Feature map

    : 입력이미지로 부터 Kernel(filter)를 사용해서 Convolution 연산으로 나온 출력 결과를 특성 맵 (feature map)이라 한다. 
    
![image](https://user-images.githubusercontent.com/59076451/128824309-273f5169-f723-4ad6-a0da-3787a0ea9135.png)

Stride 

    : Kernel의 이동 범위를 Stride라 한다. 

Padding 

    : Convolution 연산의 결과로 얻은 feature map은 원래 입력보다 크기가 작아진다는 특징이 있다. 
    
    만일 Convolution layer를 여러 겹 쌓게 된다면 출력 결과는 매우 작아질 것이다.
    
    따라서 Convolution 연산 후에도 feature map의 크기가 입력의 크기와 동일하게 하고 싶다면 padding을 사용하자.
     
![image](https://user-images.githubusercontent.com/59076451/128824593-c1ade2b7-5905-4240-873c-fc7d9c4fa204.png)

---

#### Weight, Bais in CNN

Weight 

    MLP로 이미지를 처리하는 것보다 CNN으로 이미지를 처리하는 것이 가중치 매개변수의 개수가 훨씬 적다. 다음 그림을 이용해서 이해할 수 있다.    
    
    즉, CNN은 MLP보다 더 적은 수의 매개변수를 사용하면서 동시에 공간적인 구조 정보도 유지하며 학습을 진행할 수 있게 해준다.
    
    MLP에서는 각 layer에서 연산을 진행한 후 비선형성을 추가하기 위해 activation function을 통과시킨다.
    
    CNN도 마찬가지로 Convolution 연산 후 activation function을 통과시킨다 (주로 ReLu, LeakyReLU 사용)
    
![image](https://user-images.githubusercontent.com/59076451/128825229-0323dedc-a293-4972-b349-43c224c6e0c1.png)
    
![image](https://user-images.githubusercontent.com/59076451/128825264-df42aa70-5861-4370-a25a-0f08dd2481ce.png)
    
---    
    
#### Pooling 

일반적으로 (합성곱 연산 + 활성화 함수) 이후에 Pooling layer를 추가한다. 

**Pooling layer에서는 feature map을 다운 샘플링해서 feature map의 크기를 줄이는 연산이 이루어진다.**

    주로 max pooling과 average pooling이 사용된다. 

    합성곱 연산과의 차이는 학습해야할 가중치가 없으며, 연산 후에 채널 수가 변하지 않는다는 점이다. 
    
    Pooling을 사용하면 feature map이 줄어들기 때문에 Weight 매개변수의 수를 줄여준다.

![image](https://user-images.githubusercontent.com/59076451/128827494-3287cb87-af37-45b7-85e4-3fb47916224c.png)


    
