# Transformer "All you need is Attention" `6 Dep 2017`

## Abstract 

Recurrent한 모델을 사용하지 않고서 Seq2Seq 구조를 어느정도 따르며 Attention만으로 모델을 고안하였다.<br>
결과적으로 현재 state of the art 모델들 보다 성능이 우수하며 *병렬화가 가능하다*는 큰 장점을 가진다.
  - Recurret 한 모델들은 Sequential한 특징 때문에 병렬화가 불가능했다.
  - Sequence가 길어질수록 병렬화의 필요성은 더욱 커진다. 
    - Sequence가 길어짐에 따라 입출력 간 단어의 간 길이가 길어질수록 Dependency 문제가 발생하는데 이는 Attention 기법을 통해 해결할 수 있었다.
      - Attention : Decoder에서 출력 단어를 예측하는 매 Step마다 Encoder에서의 전체 입력 정보를 한 번 더 참고하는 방법<br> 다만 전체 입력 정보를 다 동일 비율로 참고하는 것이 아니라 해당 시점에서 특별히 더 주목해야할 부분을 더 집중해서 참고하게 된다. 

## Introduction 

Transformer는 Sequential한 데이터를 처리하는 기존 모델들의 병렬화 문제를 해결하기 위해 고안되었다.<br>
NLP에서 큰 도약이었던 Attention 기법 조차 모두 Recurrent한 모델들과 같이 사용되고 있었다.<br>
  - 기존 Recurrent한 모델들은 일반적으로 입출력의 Sequence의 특정 위치에 따라 계산을 분해하여 진행한다.
  - 즉, 재귀 모델은 이전 정보들을 모두 사용하여 학습을 진행하기 때문에 Training과정에서 병렬화가 불가능하다.
  - Sequence가 길어질수록 메모리 제약, 연산량 등에서 더욱 Critical한 문제가 된다.<br>최근 연구들이 이 성능을 많이 발전시켰지만 여전히 Sequence 연산에 대한 병렬화 문제는 남아있다.
  - Attention 기법은 입출력 간 Seuqence 길이에 따른 의존성 문제를 해결한다. 
  - Attention 기법은 모두 재귀 모델과 함께 사용된다.

#### 해당 논문에서는 이 재귀적 모델을 배제하고 입출력 간의 Global한 Dependency를 모델링 할 수 있게 Attention 기법만을 사용한 Transformer를 제안
  - Transformer는 병렬화를 가능하게 하며 실제 Tranining에 걸리는 시간도 훨씬 적다.

## Background

Sequential한 연산에 대한 성능을 높이는 연구는 계속 있어왔고, 이를 시도한 모델들은 모두 은닉 상태와 입출력 연산 모두에 CNN을 Building Block으로 사용했다.<br>
하지만 이 모델들에서는 입출력 간의 두 위치로부터 신호를 관계시키기 위해 필요한 계산 수가 그 위치 간 거리에 따라 크게 증가한다.<br>
이는 곧 Dependency를 학습하는데 문제를 야기하게 된다. 

반면에 Transformer에서는 상수 번의 계산 만이 필요하다. 비록 Attention Weight가 상대적으로 평활화되어 해상도(Resolution)가 감소되는 문제가 있지만, 이는 Multi-Head Attetion기법을 사용하여 상쇄할 수 있다. 
  - 이 부분은 뒤에 조금 더 자세하게 이야기한다.

"Self-Attention" 기법은 한 Sequence의 Representation을 연산하기 위해 각기 다른 위치에 있는 요소들을 관련 짓는 Attention 기법이다. 
  - 이후 Attention 기법에 대해 조금 더 자세하게 이야기할 것.


## Model Architecture

### 1. Encoder and Decoder Stacks

![111](https://user-images.githubusercontent.com/59076451/125040637-db2f5b80-e0d2-11eb-929a-abacfb661a7c.PNG)

우선 Transformer는 Seq2Seq의 Encoder-Decoder 구조를 사용한다. <br>
Encoder는 입력 시퀀스의 연속적인 표현인 x1, x2, x3, ... , xn을 다른 연속적인 표현인 z1, z2, z3, ...., zn으로 Mapping한다. <br>
이 z 를 가지고 Decoder는 출력 시퀀스인 y1, y2, y3, ... , yn을 생성한다. <br>
각 Step에서 다음 Representation을 생성할 때, 모두 이전 정보들을 추가적인 입력으로 사용하기 때문에 Auto-Regressive하다.

Encoder와 Decoder를 각각 내부적으로 Self-Attention과 Position-Wise FC layer를 쌓아서 구성한다.

- Encoder
 
총 6개의 Encoder layer를 쌓아올려 구성한다. 각 layer들은 2개의 Sub-layer로 구성된다.<br>
첫 번째 Sub-layer는 Multi-Head Self-Attention layer이며 두 번째 Sub-layer는 Position-Wise FC feed-forward layer이다. 

또한 추가적으로 각 Sub-layer마다 Residual Connection을 차용하였고, 이무 정규화가 뒤따르도록 구성한다.

- Decoder

동일하게 6개의 Layer를 쌓아서 구성한다. <br>
각 Layer는 Encoder와 동일한 2개의 Sub-layer사이에 Encoder 출력에 대해 Multi-Head Self-Attention을 수행하는 Sub-layer를 추가배치한다. <br>
또한 은닉 상태 입력단에서 Masking 기법을 사용하는데, 이는 해당 Time step를 기준으로 미래에 해당하는 정보들의 접근을 막기 위해 사용된다. <br>
    - Masking : Decoder의 Auto-Regressive한 성질을 보존하기 위해 Leftward로의 정보흐름을 막아야한다. (해당 Step 기준으로 오른쪽 정보들의 유입)
    - 이는 해당 시점 기준 미래 정보들을 미리 조회함에 따라 현재 단어 결정에 영향을 미칠 수 있는 위치에 해당하는 값들에 -inf 값을 주어 Softmax 연산 결과 0에 가까운 값을 갖도록 하는 방식으로 Masking을 진행한다.

### 2. Attention 

#### Attention ? --- 참고 (설명은 선택)

문장을 번역할 때 I = 나 , Cat = 고양이와 같이 특정 단어에 Attention하여 하나씩 번역한다.
Attention 기법이 등장하기 전까지는 이 Alignment를 수작업으로 지정해주었지만, 이제는 자동으로 Mapping이 가능하다.
  - 단어의 대응 관계를 나타내는 정보를 Alignment라 한다.

**Dot - Product Attention**

![1112](https://user-images.githubusercontent.com/59076451/125043894-62320300-e0d6-11eb-981b-fbcd35c61e50.png)

위 그림에 따라 설명하면 , hs는 입력된 모든 단어를 나타내는 벡터들의 집합이다.

![2333](https://user-images.githubusercontent.com/59076451/125044085-96a5bf00-e0d6-11eb-9ffd-5129e4a7fe9b.png)

기존의 Seq2Se2 모델은 위 그림과 같이 인코더의 마지막 은닉 상태 벡터만을 받아서 학습했다. <br> Attention 기법은 위 hs 행렬의 마지막 행이 아닌 모든 행렬을 받아오는 것으로 생각할 수 있다.<br> Attention 기법의 핵심은 특정 단어와 대응 관계가 있는 정보를 골라내는 것이다.

![55455](https://user-images.githubusercontent.com/59076451/125044445-edab9400-e0d6-11eb-8755-1e4a099c4db8.png)

이제 위 그림과 같이 모든 행렬을 디코더의 입력으로 넣어준다. <br>
추가적으로 "어떤 계산" 이라는 Cell이 추가 되는데 이 부분에서 Attention의 핵심 연산이 일어난다. 즉, 특정 단어에 더 주목할 수 있도록하는 연산이다.

특정 단어에 주목할 수 있도록 하는 방법은 다음과 같다. 

![aa](https://user-images.githubusercontent.com/59076451/125044727-36634d00-e0d7-11eb-92f3-011462e759ac.png)

a는 각 단어의 중요도를 나타내는 가중치이다. a는 0~1 사이의 값으로 구성되어있으며 아래와 같이 hs와 곱하여 가중합을 구하는데에 사용된다.

![bb](https://user-images.githubusercontent.com/59076451/125044860-598dfc80-e0d7-11eb-8923-cb1aa5810d94.png)

가중합이란 대응하는 원소끼리 곱한 뒤 결과를 모두 합하는 계산인데, 이에 대한 결과로 우리는 Context Vector를 얻는다. <br>
위 그림에서 '나' 에 대한 가중치가 0.8인데 이에 결과로 나온 Context Vector는 '나' 라는 단어에 대한 정보를 더 많이 담고 있다. <br>
  - 논문에서는 Context Vector를 Attention Value라 이름 붙이고 있다.

이제 위의 가중치 합을 구하기 위해 먼저 알아야할 가중치 a를 구하는 방법이다.

![666](https://user-images.githubusercontent.com/59076451/125045308-d0c39080-e0d7-11eb-81f4-899a9f8b0af2.png)

디코더의 LSTM 은닉 상태 벡터 h가 hs의 각 vector와 얼마나 유사한지를 찾아내는 과정이다.<br>
이를 파악하기 위한 여러가지 방법이 있지만 가장 간단한 '내적'을 이용한다. <br> 

**Dot-Product Attention 기법 **
벡터 내적을 통해 두 벡터가 얼마나 같은 방향을 바라보고 있는지 판단할 수 있다.

![777](https://user-images.githubusercontent.com/59076451/125045584-1da76700-e0d8-11eb-8433-7900dbf653e4.png)

벡터 내적을 통해 위와 같이 유사도를 구한다. S를 보면 첫 번째 원소 '나'에대한 유사도가 0.9로 가장 높다. <br>
여기서 S를 Attention Score라 한다. <br> 다음으로 이 s를 0~1사이의 값으로 정규화한다.

![888](https://user-images.githubusercontent.com/59076451/125045750-4d566f00-e0d8-11eb-9b6e-5474db45e46f.png)

Softmax 함수를 사용하여 S를 정규화하면 가중치 a를 구할 수 있다. 

![ffff](https://user-images.githubusercontent.com/59076451/125045905-7545d280-e0d8-11eb-8a60-725ae0a2f17d.png) ![jjj](https://user-images.githubusercontent.com/59076451/125045966-842c8500-e0d8-11eb-858e-d6751bd21e0f.png)

이제 위 과정들을 도식화하면 위와 같다. 이 계산을 수행하는 층을 Attention layer라 한다. <br> 이 계층을 통해 인코더가 출력한 hs에서 중요한 원소에 주목하며, 이를 기반으로 Context vector를 구해 윗 단으로 올려보낸다.

바로 이 부분이 위의 "어떤 계산"에 해당하는 Layer이다.

### 다시 논문 

![222](https://user-images.githubusercontent.com/59076451/125042647-09ae3600-e0d5-11eb-9a32-4bf2ea489e4f.PNG)

Attention 기법은 Query , Key, Value들이 모두 벡터일 때, Query와 Key-Value 쌍의 집합을 출력에 Mapping하는 것<br>
**즉, 디코더의 특정 시점에서 출력 단어를 예측할 때마다 인코더에서의 전체 입력 정보를 참고한다는 것. <br>**
**다만 해당 시점에서 예측해야할 단어와 관련이 깊은 정보에 더 집중하도록 함.**

- Q : h
- key : hs
- Value : a


하지만 Transformer 논문에서는 Q , K, V를 Vector로써 매 Time Step마다 Sequential하게 사용하는 재귀 모델과는 다르게 Matrix로 Packing 하여 사용한다.
이를 Multi-head Attention이라 부른다.

위에서 Attention Weight가 상대적으로 평활화되어 해상도(Resolution)가 감소되는 문제를 이야기했고 이를 Multi-Head Attention 기법으로 해결했다고 말한다. <br>
이에 대한 개인적인 이해는 다음과 같다.
  - 전체 모델 Sequence는 임베딩을 통해 512차원으로 사용한다.
  - 기존 재귀모델에서 Single Attention을 사용하여 학습하게 되면, 각 Step에서의 단어마다 Attention을 구해 가중치의 해상도가 좋다.
  - 하지만 Transformer와 같이 Q,K,V를 행렬로 Packing하여 모든 단어에 대한 Attention을 구하게 되면, 단어마다의 Attention의 해상도는 상대적으로 평활화되게 된다.
    - 가중치는 결국 0~1 사이의 값을 가지게 되므로, 표현하려는 영역이 넓어질수록 그 Resolution이 떨어질 것 
  
Multi-Head Attention은 위와 같은 해상도 평활화 문제를 어느 정도 억제해준다고 한다. 이유와 방법은 다음과 같다.
  - 논문에서는 전체 모델 Sequence 512개를 8등분하여 각 8개의 다른 Sequence Domain에서 Attention을 진행한다.
  - 이 후 이를 Concate하여 선형 변환을 거쳐 최종 결과를 도출한다.
    - 즉, 1개의 큰 덩어리를 8개의 Sub-space로 나누어 Attention을 적용하는 방식으로 Resolution을 높인다.
    - 기존 재귀 모델과 다르게 Transformer는 병렬적으로 처리가 가능하므로 이또한 상당히 좋은 아이디어이다.

#### Scaled Dot-Product Attention

![hhhhssd](https://user-images.githubusercontent.com/59076451/125048675-2188b880-e0db-11eb-9a3e-918395140885.PNG)

해당 논문은 흔히 사용되는 Attention 함수 중 dot-product attention을 사용했다. 여기서 scaling factor로써 1/sqrt(dk)가 추가 되었다.<br>
연구진들은 dk가 커질 수록 dot product의 결과값도 커지는 경향이 있어, 결과적으로 softmax 함수에서 gradient가 매우 것을 방지하기 위해 Scaling을 진행하였다고 한다.

- dq , dk, dv : Query, Key, Value의 차원


#### Application of Attention in our Model

Transformer는 multi-head attetion을 3가지 다른 방식으로 사용한다.

- Encoder-Decoder attention 층에 입력되는 쿼리들은 이전 Decoder layer의 출력으로부터 오고, 키와 벨류들은 Encoder의 최종 출력에서 온다.<br> 이는 Decoder의 모든 위치에서 입력 시퀀스의 모든 위치에 접근할 수 있음을 허용한다.
- 
![1](https://user-images.githubusercontent.com/59076451/125050489-f606cd80-e0dc-11eb-9a89-e58b93585f4a.PNG)

- Encoder 는 Self-attention layer를 포함한다. Self-attention layer에 입력되는 모든 key,value,query들은 하나의 시퀀스에서 온다.(이전 인코터 층의 출력)<br>따라서 각 Encoder step에서 이전의 모든 Encoding 위치에 접근할 수 있다.

![2](https://user-images.githubusercontent.com/59076451/125050493-f69f6400-e0dc-11eb-96fe-56ee048cd003.PNG)

- Decoder 내부의 Self-Attention layer는 위와 비슷하게 이전의 모든 Decoding 위치에 접근할 수 있다. <br> 다만 디코더의 Auto-gressive 성질을 보존하기 위해 Masking 기법을 추가한다.

![3](https://user-images.githubusercontent.com/59076451/125050495-f737fa80-e0dc-11eb-9aa7-878d60859b50.PNG)


### 3. Position-Wise Feed-Forward Networks

Attention Sub-layer에 더해서, Encoder와 Decoder의 각각의 Position에 대해 독립적으로 FC feed-forward network를 포함한다. 
이 layer는 두 번의 선형변환과 그 사이의 ReLu 활성함수를 거치도록 구성한다. 

![hhhhssd](https://user-images.githubusercontent.com/59076451/125050767-40884a00-e0dd-11eb-928f-284d3f4bdf79.PNG)

### 4. Embeddings and Softmax

input과 output 토큰들을 d_model 차원 벡터로 변환하기 위해 Embedding을 사용한다.<br> 
또한, decoder의 output으로 다음 Step의 represention을 예측하기위해 학습가능한 선형 변환과 softmax 활성함수를 사용하였다.<br> 
우리는 두 임베딩 층과 softmax 이전의 선형변환에 동일한 가중치행렬을 공유하여 사용하였다. 임베딩 층들에서는 그 가중치들에 sqrt(d_ model)를 곱하여 사용하였다.



#### 참고 자료 
1. https://velog.io/@changdaeoh/Transformer-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0
2. https://bkshin.tistory.com/entry/NLP-14-%EC%96%B4%ED%85%90%EC%85%98Attention



