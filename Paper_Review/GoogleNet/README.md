# GoogleNet - Going Deeper with Convolutions `sep 17 2014`

*논문 순서와 별개로 이해하기 편한 흐름으로 정리하였음*

## Abstract

GoogleNet은 기존의 CNN 구조를 향상시키기 위해 고안된 새로운 모델이다.<br>단순히 CNN 성능을 높이는 방법은 다음과 같다. 
  
  1. depth를 늘리는 것
  2. 각 layer의 unit/cell을 늘리는 것

당연하게도 각 방법에 대한 부작용이 존재한다.

  - For 1 method : Parameter수가 많아지기에 제한된 데이터에 대해서 overfitting의 위험도가 증가한다. <br>
  - For 2 method : Computing power가 증가하게 된다. 
  
  ex) filter수가 늘어나게 되면 연산량은 그에 대해 Quadratic 하게 증가한다.
  
##### GoogleNet은 여기에 따른 해결책으로 (depth를 늘리고, unit을 늘리며 연산량은 유지할 수 있는) Inception Module을 포함한 모델을 제시한다.
  
GoogleNet의 주요한 특징은 다음의 두 가지로 볼 수 있다.

  1. hebbian Principle (1x1 Conv layer)
  2. Multi - scale Processing (Inception Module)

<br><br>

## Network in Network , 2013 (NIN) 

GoogleNet은 NIN 논문에서 시작한다.

NIN 논문에서는 Conv layer가 local receptive field에서 특징을 뽑아내는 능력은 좋지만, 정작 filter는 linear하기 때문에 기껏 뽑아 놓은 비선형적 특징을 많이 걸러내버리는 문제가 있다고 말한다.

따라서 이 문제를 해결하기 위해 filter의 개수를 늘려서 비선형적인 특징을 더 많이 뽑아내고 걸러지고 나서 남는 feature가 많도록 해야 한다고 생각한다.<br>하지만 당연히 연산량의 문제로 마냥 filter를 늘릴 수는 없었기 떄문에 다른 방법을 고안한다.

  - NIN 논문에서는 해결책으로 Micro Neural Network를 제시한다. 
  ![fasdfasdfasd](https://user-images.githubusercontent.com/59076451/118940209-70dc2380-b98b-11eb-9ec7-11d05aa31ba2.PNG)
  
    - 다음 그림은 각 layer에서 Convolution을 filter가 아니라 MLP를 사용하여 각 Patch를 Swapping하며 feature를 추출하도록 한 것이다.
    - 위와 같은 방법으로 비선형적인 특징을 더 잘 뽑아내게 되었다.
    
    - 하지만 연산량이 늘어나는 문제는 여전히 발생한다.    
      - NIN은 여기에 1x1 Convolution을 이용하여 연산량을 줄이는 방법을 제시한다.

여기서 GoogleNet은 Micro Neural Network를 CNN에 더 알맞게 적용해보기 위해 Inception Module을 고안한다. 

##### 즉, GoogleNet에서 눈에 띄는 특징인 Inception Module과 1x1 Conv layer를 이용한 차원축소/bottleNet 방법은 NIN에서 출발하였음을 알 수 있다.

* 1x1 Conv 

  1x1 Conv 는 차원을 줄이기 위해 사용된다. <br> 이를 이용하여 여러 개의 feature map으로부터 비슷한 성질을 갖는 것들을 묶어낼 수 있다. (Encoder 느낌으로 이해하였다 -> 그려서 같이 설명할 것)
  
    - CCCP (Cascade Cross Channel Pooling) : Channel을 직렬러 묶어 픽셀 별로 Pooling을 수행하는 것 
      - 결과적으로 Feature map의 크기는 그대로, Channel수만 줄어들어 차원 축소의 효과를 낸다.
      
       ![12312](https://user-images.githubusercontent.com/59076451/118950056-cec13900-b994-11eb-9279-6e14f4b56024.PNG)
    
      - Channel 4 Feature map -> Channel 2 Feature map 
       
       ![ghjgjhjgh](https://user-images.githubusercontent.com/59076451/118941810-15129a00-b98d-11eb-9fbb-bd7845715e3f.PNG)

추가적으로 GoogleNet의 말단 부분은 FC layer 대신에 Global Average Pooling을 사용한다. 이 또한 NIN 논문에서 차용한 것으로 보이는 아이디어이다.

NIN에서는 모델 앞단에서 효과적으로 feature vector들을 추출하였기 때문에 이런 vector들에 대한 pooling만으로도 충분하다 라고 한다.<br>
FC layer는 모델 전체 free 파라미터의 90%에 달하기 떄문에 overfitting에 빠질 가능성이 매우 높고, Dropout 기법을 사용해야 한다.

하지만 Average pooling 만으로 분류 classifier 역할을 할 수 있기 때문에 overfitting 문제도 피할수 있고, 연산량도 대폭 줄일 수 있다는 장점을 적극 어필한다.


## Motivation and High Level Considerations

GoogleNet의 CNN의 성능을 높이는 두 가지 방법을 유지하면서, 그 단점을 보완하기 위해서 시작되었다.<br>
즉, NIN의 Mirco Neural Network를 차용하여 Inception module을 구상하였고, Computing Power를 일정량으로 유지하기 위해 1x1 Conv 를 적극 사용한다.

하지만 추가적인 동기가 있다.<br>
CNN의 성능을 높이는 두 가지 방법에 대한 가장 기본적인 해결책은 NIN의 논문에서 가져온 아이디어가 아니라 Dense하게 연결된 구조를 Sparse하게 연결된 구조로 만들어 CNN을 구성하는 것이다.

  - 왼쪽이 Sparse한 layer이고 오른쪽이 Dense한 layer이다.
  ![werwqe](https://user-images.githubusercontent.com/59076451/118943709-d4b41b80-b98e-11eb-9864-bf62815e3c2b.PNG)
  
하지만 현재 Dense한 구조에 대해서는 컴퓨터가 아주 좋은 성능을 보이지만 Sparse한 구조에 대해서는 그보다 훨씬 못한 성능을 보인다고 한다.
CNN 또한 잠시 Sparse한 구조로 도전을 했지만, 바로 고개를 돌려서 다시 Dense한 구조로 돌아왔다고 첨언한다.

Sparse하게 연결된 모델 구조를 제안한 논문에서는, 통계적으로 각 layer에 대해 출력과 가장 연관성이 높은 Unit들만 골라내서 연결하면 좋은 성능을 낼 수 있지 않겠냐고 제안한다.<br>
(Node/Cell/Unit이 선택될 확률을 더 Sparse 하게 하면서 더 깊은 신경망으로 만들고, 입력층에서 출력층으로 이어지는 layer의 Unit간의 관계를 통계적으로 잘 분석해서 입출력 간의 관계가 높은 Unit들만 골라낸다면. <br> 그리고 그 Unit들로 이루어진 Sub Dense layer들을 만들 수 있다면 최적의 Sparse한 모델을 만들 수 있다고 주장한다.)

![KakaoTalk_20210520_172102394](https://user-images.githubusercontent.com/59076451/118944873-ea761080-b98f-11eb-95dc-12b697ce02be.png)

![KakaoTalk_20210520_172116599](https://user-images.githubusercontent.com/59076451/118944906-f366e200-b98f-11eb-8a48-c7de5d45ef82.png)

GoogleNet은 이 Sparse한 구조에 대해서도 이 아이디어로 무언가 해보려고 했는데, 이 생각의 결과가 Inception Module이다.

*Inception Module은 이런 Sparse 구조에 내놓은 해결책의 성능을 시험해보기 위해 시작되었다고 한다.* <br>
*실제로 하이퍼 파라미터를 조정하며 구현한 결과 꽤 좋은 성능을 내었다고 한다*.

즉, 서로 연관성이 있는 Unit들을 뽑아내는 것 그리고, 그 Unit들로 작은 Sub dense layer를 만드는 것에 주목했는데 이것이 논문 Abstract에서 이야기하는 multi-scale processing이다.

정리하자면 Inception 구조의 주 아이디어는 기존의 CNN 구조의 각 layer에서 입출력에 대해 가장 연관성이 높은 Unit들 선택하여 최적의 Sparse Data Unit을 모아 Sub Dense한 구조를 만드는 것이다.

*여기서 입력단에서 가장 가까운 layer에서는 특징이 한 Unit에 몰릴 가능성이 높은데, 이 때는 1x1 Conv 를 통해 (PCA와 같이) 가장 주요한 feature 하나만 추출하도록 하여 불필요한 연산을 줄이는 것 같다.* 

![fghjfgjh](https://user-images.githubusercontent.com/59076451/118947170-0975a200-b992-11eb-970f-47985bd13042.PNG)

위 그림으로 우리가 연관성이 높은 Unit을 고른다는 말에 대한 Insight를 얻을 수 있다.

만약 다양한 영역의 local recpetive field를 다룬다면 우리는 Sparse하게 뿌려진 feature data에 대해서 서로 관계가 있는 Unit들을 선택할 수 있는 확률이 높아진다.<br>
작은 patch로 이미지를 살피고, 또 큰 patch로 이미지를 살피는 직관적인 이유는 다음 예를 통해 이해한다.

- 만일 위의 세 번째 그림과 같이 작은 patch 만으로 그림을 살핀다면 그 patch에 포함된 선이 실제로는 원을 구성하는 일부라는 unit간의 관계성을 놓치게 된다.
- 이를 위해 네 번째 그림과 같이 더 큰 patch인 filter도 이용한다.
  
  - 즉, 작은 patch와 더불어 큰 patch를 이용하면 각 unit의 상관관계를 유지하며 feature map을 구성할 수 있다고 이해할 수 있겠다.

따라서 Inception Module은 아래 그림과 같이 다양한 크기의 conv filter를 사용하여 Module block을 구성한다. 
![gjhfghfghd](https://user-images.githubusercontent.com/59076451/118947188-0ed2ec80-b992-11eb-96ed-186977ee21eb.PNG)

하지만 위의 Inception module block을 그대로 사용하면 당연히 filter를 여러개 사용하는 것이기 때문에 연산량의 문제가 발생한다.

![ghfghfgsdh](https://user-images.githubusercontent.com/59076451/118947916-c5cf6800-b992-11eb-8a15-52c3c66c112c.PNG)

따라서 위의 그림과 같이 1x1, 3x3, 5x5 filter를 그대로 병렬적으로 사용하되 3x3, 5x5 filter에는 직렬적으로 1x1 conv filter를 사용하여 차원을 줄여 연산량을 감소시킨다.


![wer2qw](https://user-images.githubusercontent.com/59076451/118948432-3aa2a200-b993-11eb-9ea5-292c5457f471.PNG)

추가적으로 depth가 깊어질수록 feature들은 더 추상적인 개념을 가지고 있는데, 이 feature를 잘 추출해 내기 위해 더 많은 conv filter들이 필요하다.<br>
따라서 깊은 layer에서는, 1x1, 3x3, 5x5 conv filter가 더 많이 사용되는 것을 볼 수 있다.

(여기서 추상적이라는 개념은 feature가 많은 특징을 포함하고 있다는 뜻이다. 즉, Encoder로 예를 들면 10개 Node -> 1개 Node로 압축하는 것보다 100개 Node -> 1개 Node로 압축할 때의 feature가 더 추상적이다라고 할 수 있겠다.)

추가적으로 효율적인 메모리 사용을 위해 낮운 layer에서는 기존 CNN 모델 구조를 사용하고, 높은 layer부터 Inception Module을 사용하는게 좋다고 한다. 


## Model architecture 


![image](https://user-images.githubusercontent.com/43063980/119124018-29c55f80-ba6b-11eb-870b-9304f1c340b1.png)
![image](https://user-images.githubusercontent.com/43063980/119124032-2cc05000-ba6b-11eb-85fd-d71fb5a4a6ed.png)
![image](https://user-images.githubusercontent.com/43063980/119124044-3053d700-ba6b-11eb-8192-ce4d3fdcd665.png)

- **구성**

1. (노랑) Inception module을 9개 쌓음
2. (빨강) Auxiliary classifier 2개 추가 
    > 3,6번째 inception module 뒤에 (softmax) classifier를 추가함.

    > 기존의 모델은 classifier가 뒷단에 하나만 존재해서 deep한 모델에서는 vanising gradient가 발생했는데 
    > Auxiliary classifier를 중간에 하나씩 추가해줌으로써 vanishing gradient를 완화시킴.
    
    > 다만 loss계산에서는 가중치는 0.3으로 합쳐짐.

3. (민트) Average Pooling 

4. 22개의 convolution layer + 5개의 pooling layer 

<br>

- **가장 잘 나왔던 GoogLeNet 정보** 
![image](https://user-images.githubusercontent.com/43063980/119125605-f388df80-ba6c-11eb-90ab-1e255c23adc0.png)

- patch size/stride : filter 크기, stride 값 
- output size : convolution 거친 후 나오는 feature map의 크기 및 개수(마지막 값)
- '#' : 개수
  
  > #3x3 : 3x3 conv 연산 후 생기는 feature map 수 
  
  > #3x3 reduce : 3x3 conv 전에 거치는 1x1 conv의 개수 
  
- pool proj : 3x3 mas pooling, lxl conv
- params : 파라미터크기
- ops : 연산크기
  
<br>
  
  
  
## Result 

ILSVRC 2014 대회는 1000개의 이미지 관련 task 평가
1.2 million images for training,50,000 for validation and 100,000 images for testing.
the top-5 error rate 를 기준으로 함.

<br>

논문에는 classification, detection 부분의 결과가 나와있음.

![image](https://user-images.githubusercontent.com/43063980/119127051-c6d5c780-ba6e-11eb-8cba-959a73dba225.png) 
![image](https://user-images.githubusercontent.com/43063980/119127068-cc331200-ba6e-11eb-9275-02508ed272cb.png)





**[참고]**

paper : https://arxiv.org/pdf/1409.4842.pdf

https://phil-baek.tistory.com/entry/3-GoogLeNet-Going-deeper-with-convolutions-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0

http://www.hellot.net/new_hellot/magazine/magazine_read.html?code=202&sub=002&idx=45531

https://m.blog.naver.com/PostView.naver?blogId=qbxlvnf11&logNo=221429203293&proxyReferer=https:%2F%2Fwww.google.com%2F

https://oi.readthedocs.io/en/latest/computer_vision/cnn/googlenet.html



