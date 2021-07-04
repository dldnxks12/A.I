# RNN, LSTM, GRU 개념 정리 및 비교


기존의 인공 신경망은 데이터의 순서나 맥락이 고려되지 않은 단순한 구조였다.  
따라서 번역이나 주가예측과 같은 Sequence data에 대한 task를 해결할 수 없었다. 

이로 인해 Sequence data를 학습하기 위해 hidden layer의 정보를 다음 단계로 넘겨주는 형태의 구조를 도입하게 되었고, 이를 통해 과거의 정보가 미래의 결과에 영향을 줄 수 있는 **순환신경망(RNN)** 이 나타났다.

하지만 순환신경망은 Sequence의 길이가 길어짐에 따라 문제가 발생한다.
학습을 거치며 이전의 정보를 쉽게 잃어버리거나(Gradient Vanishing), 이전 정보에 대한 가중치가 거대해져(Gradient Exploding) 나머지 데이터들을 무의미하게 만들어버리는 등의 문제가 발생한다.  

이 Gradient Vanishing, Gradient Exploding 문제를 해결하기 위한 방법으로써 앞단의 정보를 알맞게 조정해가며 사용할 수 있고, Computing면에서도 더 효율적으로 발전한 것이 **LSTM**과 **GRU**이다. 

<br>

## 1. RNN (Recurrent Neural Network)

![111](https://user-images.githubusercontent.com/59076451/123301068-2e8f9e80-d556-11eb-94e1-dbbd12119ac0.PNG)

RNN의 기본 컨셉은 이전 정보를 다음 Step으로 넘겨주어 Context 정보를 유지하도록 하는 것이다. 
예를 들어 "I work at google." 과 "I google at work." 는 완전히 다른 의미이며, 이러한 문맥 정보를 죽이지 않고서 학습할 수 있도록 만든 모델인 것이다.

한 가지 오해할 수 있는 것은 다음 Step에서 바로 직전 Step의 정보만을 이용하는 것이 아니라 해당 Step 이전의 모든 정보를 사용한다는 것이다.

RNN의 특정 Step에서 기본 학습 과정을 먼저 살펴보자.

![4444](https://user-images.githubusercontent.com/59076451/123301125-423b0500-d556-11eb-85e6-34c84efc5f32.png)

**x_t** : 입력 데이터<br>**h_t-1** : 이전 시점에서 넘겨준 문맥정보<br>**h_t** : 이전 정보와 현재 입력 데이터를 이용한 정보<br>**y^_t** : h_t를 통해 얻은 해당 Step에서의 Hypothesis<br>**y_t** : Target Value

그리고 위 그림에서 색이 입혀진 **W**는 각각 입력데이터, 이전 문맥 정보, 출력 정보에 대한 가중치이다.
- **Whh Wxh Wyh**

#### **I Work at Google**을 예시로 들어보자.

위 문장을 먼저 각 문장 요소로 분리한다. (Target Value : 대명사, 명사, 동사, 전치사)

1. 먼저 입력 데이터로  **I**를 넣어주면 해당 step 이전의 정보가 없기 때문에 h_t-1은 0으로, 초기 가중치 **Whh Wxh Wyh**는 랜덤한 값으로 초기화하여 학습을 시작한다.
2. 입력 데이터에 대해서 선형 회귀를 거쳐 tanh 연산을 수행한다.        
3. 해당 연산 결과는 현재 Step의 출력으로도 사용되며 다음 Step으로 넘겨줄 데이터로도 사용된다.
4. 해당 데이터는 다시 한 번 선형 회귀 거쳐 Softmax 활성함수를 통해 결과를 출력한다.
5. 해당 결과와 Target Value를 비교하여 Loss를 계산하고 이를 줄이는 방향으로 학습한다.

6. 다음으로 **Work** 데이터를 입력한다. 이 데이터는 위와 동일한 과정을 거치지만, 이전 데이터 **I**의 정보가 남아 학습에 영향을 주게 된다.
7. 다음으로 **at** 데이터를 입력한다. 이 데이터는 **I** 정보의 영향을 받은 **Work**의 영향을 받으며 학습이 진행된다.

위 1-7 번과 같은 과정으로 RNN 구조의 학습이 진행된다고 이해할 수 있다.

#### 조금 더 디테일한 부분을 살펴보자

입력 데이터와 출력 데이터, Target Value의 형태는 모두 Vector이다. 
물론 방금 언급한 Target Value 역시 명사, 대명사, 전치사, 동사 모두 Vector 형태로 표현되어 학습에 사용된다.


![2222](https://user-images.githubusercontent.com/59076451/123303359-af4f9a00-d558-11eb-88ad-824c2ade1e3d.PNG)

위 그림은 **Hell** 을 넣었을 때 **o**가 나오도록 학습하는 과정이다.<br>
각 Character 들 모두 각각 Vector 형태로 표현되고 출력 또한 Vector 형태이다.<br>
출력은 최종적으로 Softmax를 통해 가장 높은 값을 가진 Index와 Target Value의 Index와 비교될 것이다.<br>
이를 통해 Loss를 계산할 수 있는데, 우리는 이렇게 각 Step에서 발생하는 Loss를 합쳐 전체 Loss를 계산한다.

여기까지의 과정이 RNN에서의 순전파 과정이다.<br>
역전파 과정은 당연히 다음과 같이 Chain Rule을 사용하여 사용한 모든 정보들에 대한 Gradient를 구해서 업데이트한다. <br>
예를 들어 **T = 3**에서의 가중치는 **T : 0 ~ 2**까지의 정보를 사용하여 얻은 결과이다. <br>
따라서 해당 **T=0, T=1, T=2** 시점의 Gradient들도 이 Loss에 대해서 Update를 해주어야 한다는 것이다.


![55555](https://user-images.githubusercontent.com/59076451/123304485-f68a5a80-d559-11eb-8c6a-131ecbc1f666.PNG)

Gradient Vanishing, Gradient Exploding 문제는 바로 이 역전파 과정에서 발생한다.
앞서 이야기한대로, 현재 T시점의 Gradient를 업데이트하기 위해 이전 정보에 대한 Gradient를 모두 계산해서 전달해야하는데 Sequence 길이가 길다면 앞쪽까지 충분히 전달하지 못하는 것이다.

즉, T시점에서 발생한 오차에 대한 Gradient를 앞단의 Cell들은 사용하지 못하는 것.


<br>


## 2. LSTM(Long Short Term Memory)

RNN은 순환구조를 통해 데이터의 연속성을 학습할 수 있게 만들었지만  
Backpropagation 과정에서 **Gradient Vanishing**이 발생한다.

RNN의 장기 의존성 (Long-Term Dependency) 문제를 **Cell state**를 추가하여 해결한 모델이 **LSTM**이다.

![image](https://user-images.githubusercontent.com/43063980/123243333-e18fd600-d51d-11eb-9904-cd90450aad9e.png)

- LSTM은 RNN과 마찬가지로 순환구조를 가지고 있다. 다만 RNN에서 사용한 **h_t** **모듈의 구조가 다르다**.  

<br>





## Cell State 


<img src = "https://user-images.githubusercontent.com/43063980/123247802-36355000-d522-11eb-88d2-f9203b6c69f3.png" width="40%">

- LSTM Cell 구조를 살펴보면 특정 루트는 과거의 정보를 잊어버리게도 만들고, 또 다른 루트는 현재의 정보를 얼마나 사용할지를 조정할 수 있도록 한다.

어떤 정보를 update시킬지는 **루트(Gate)**를 통해 결정된다.


<br>


## Gate

![image](https://user-images.githubusercontent.com/43063980/123250826-7b0eb600-d525-11eb-8114-dee99dad6b7d.png)

- 각각의 그림은 **Forget gate**, **Input gate**, **Output gate**를 나타낸다.

    - Forget gate : 과거정보를 얼마나 잊을 것인지
    - Input gate  : 현재정보를 얼마나 기억할 것인지
    - Output gate : 다음 state로 보낼 output 결정
 
 - 이전 cell state는 3개의 gate를 거처 다음 cell state로 넘어간다.
 - 모든 gate는 sigmoid함수를 사용하여 cell state에 얼만큼 영향을 줄지 결정한다.
 
       0.0X : 정보기억↑↑  
       0.9X : 정보기억↓↓
 
 즉, 이전 RNN에서는 단순히 이전 정보(**h_t**)와 입력 데이터(**x_t**)를 받아 학습에 사용했다면, LSTM은 이전 정보와 입력 정보를 가공해서 사용한다고 생각할 수 있겠다.

<br>


<br>

> **Forget gate**
<img src = "https://user-images.githubusercontent.com/43063980/123253320-536d1d00-d528-11eb-8879-b73b36636c45.png" width="30%">

> **Input gate**
<img src = "https://user-images.githubusercontent.com/43063980/123253709-b9f23b00-d528-11eb-80b9-52db0f3cdc01.png" width="30%">

- Forget gate와 Input gate의 식은 위와 같다.
    - 각각 이전 데이터와 입력 데이터를 선형 회귀 한 후 Sigmoid 활성 함수를 통해 해당 결과를 얼마나 사용할 지 결정한다.
- 이들은 아래의 수식을 통해 cell state를 update한다. 

<br>

> **update** 

<img src = "https://user-images.githubusercontent.com/43063980/123258539-71d61700-d52e-11eb-910f-472f192283f7.png" width="40%">

<img src = "https://user-images.githubusercontent.com/43063980/123258513-68e54580-d52e-11eb-8767-7fd9a2823f6a.png" width="30%">



- input gate의 ~Ct를 보면 원래 RNN의 식과 동일하다. 
- 이전의 정보(Ct-1)와 forget gate를 연산하고 이번 cell에 대한 값(~Ct)은 input gate와 계산한다.   
- 과거정보와 현재정보가 조정되어 합쳐진 cell state는 **다음 state로 넘어간다.**  


<br>

> **output gate**
<img src = "https://user-images.githubusercontent.com/43063980/123253790-d3938280-d528-11eb-8881-74813b49656f.png" width="30%">


- 최종적으로 얻어진 cell state 값을 얼마나 hidden state로 넘겨줄지 결정하는 역할
- cell state는 output gate를 거쳐 hidden state로 넘어간다.  

<br>

**Cell State**는 일종의 컨베이어 벨트와 같은 역할을 하며 정보가 잘 흐르도록 도와준다. <br>덕분에 Sequence가 길어지더라도 Gradient가 비교적 잘 전파된다.

RNN에서는 앞 단의 정보들이 긴 시간 동안 유지되지 못했던 반면, LSTM은 컨베이터 벨트와 같은 구조로 필요한 정보만 선별하여 넘겨주기 때문에 Long-term dependancy 문제가 없다고 한다.

**RNN의 문제였던 장기의존성 문제를 cell state라는 레이어를 통해 해결했지만 다른 RNN계열보다 연산속도가 느리다는 단점이 있다.**

<br>

<br>


## 2. GRU(Gated Recurrnet Unit)
GRU도 마찬가지로 순환구조를 가지고 있다. 이 역시 모듈의 구조가 다르다.  
GRU는 더 간단한 구조로 이루어져 있어서 계산이 효율적이다. **(연산속도를 높였다.)**


<br>

**[LSTM과 비교]**
- LSTM에 비해 학습속도가 빠르다.
- 데이터가 적을 때, 좋은 성능을 보인다. (데이터가 많은 때는 LSTM의 성능이 더 좋다.)
- reset gate, update gate 총 2개의 gate가 사용된다.

<br>


## Gate
LSTM의 Input Gate와 Forget Gate가 GRU에서는 하나의 Update Gate로 합쳐졌다.  
LSTM의 cell state와 hidden state가 GRU에서는 하나의 hidden state로 합쳐졌다.


<img src = "https://user-images.githubusercontent.com/43063980/123234349-d5a01600-d515-11eb-8071-6aceac6b2ec4.png" width="50%">

<br>


> reset gate
<img src = "https://user-images.githubusercontent.com/43063980/123263212-b0ba9b80-d533-11eb-8799-84b6af4ea60d.png" width="30%">

- 이전의 hidden state를 얼마나 활용할지

> update gate


<img src = "https://user-images.githubusercontent.com/43063980/123276530-36dcdf00-d540-11eb-9b04-39a0c246ac92.png" width="30%">

- Zt(controller)가 동시에 forget과 input gate를 모두 제어한다

    > Zt : 현재정보를 얼마나 사용할지 (input gate)  
(1-Zt) : 과거정보를 얼마나 사용할지 (forget gate)



<br>


<br>


> update

<img src = "https://user-images.githubusercontent.com/43063980/123278831-3f361980-d542-11eb-8881-05f554cd58d3.png" width="30%">

- 현재에 대한 hidden state를 구하는 식
- 이전 hidden state을 얼마나 사용할지 Reset gate를 적용하여 현재 hidden state로 사용할 값을 구한다.



<img src = "https://user-images.githubusercontent.com/43063980/123278567-01d18c00-d542-11eb-9155-6d038360b208.png" width="30%">

- 과거의 hidden state와 현재의 hidden state를 각각의 비율(Zt, (1-Zt))에 따라 반영하는 식
- update gate를 거쳐 다음으로 hidden state 값을 넘겨준다.



<br>

-> **연산속도가 느리다는 LSTM의 단점를 모듈구조를 간결하게 만듦으로서 계산을 효율적이게 만들었다.**



#### 참고 
https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=winddori2002&logNo=221992543837
https://wooono.tistory.com/242
https://blog.naver.com/PostView.nhn?blogId=winddori2002&logNo=221974391796
https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/

