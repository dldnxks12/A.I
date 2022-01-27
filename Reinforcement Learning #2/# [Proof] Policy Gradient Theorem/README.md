### Policy Gradient Theorem 증명

PG Base Algorithm의 충분한 이해를 위함..

- Reference 

[1] SuttonBarto_2nd_pdf   
[2] https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html   
[3] https://talkingaboutme.tistory.com/entry/RL-Policy-Gradient-Algorithms

---

<br>

- `Objective Function`

       모델을 평가하는 방법 1. Value function 2. Action Value function

       PG의 목적은 GD Method를 통해 모델을 향상시키는 것이므로 목적함수가 Value , Action Value function의 형태.


<div align="center">

Continuous State에 대한 Objective Function

![img.png](img.png)

Episodic한 경우 Start Value를 통해 Objective Function을 정의할 수도 있다.

![img_1.png](img_1.png)

위 두가지 형태 모두 동일한 Gradient 결과를 반환한다. (이후 증명)

</div>

<br>

- `Policy Gradient Theorem` 

<div align="center">

PG Theroem은 GD Method에 필요한 `Objective function의 미분`이 다음과 같은 형태로 나타남을 의미한다. 

![img_2.png](img_2.png)


</div>

<br>

- `Stationary State Distribution & Visitation Probability`

<div align="center">

![img_3.png](img_3.png)

</div>

<br>

- `Proof of PG Theorem`

<div align="center">


![img_4.png](img_4.png)

![img_5.png](img_5.png)

![img_6.png](img_6.png)

![img_7.png](img_7.png)

![img_8.png](img_8.png)

</div>