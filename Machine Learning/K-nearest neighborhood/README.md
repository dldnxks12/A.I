## `K-nearest neighborhood`

`Supervised learning 중 가장 직관적이고 고전적인 방법`


`K-nearest neighborhood algorithm`은 지도 학습 알고리즘 중 하나로, `거리 기반 분류 모델`이다


---

#### K-nearest neighborhood algorithm

    최근접 이웃 알고리즘이라고 부르는 이 방법은 새로운 데이터가 주어졌을 때, 어떠한 데이터들과 가장 닮아있는지를 찾는다.
    

<div align=center>
  
![image](https://user-images.githubusercontent.com/59076451/132224965-98b7a129-b8fd-4165-9aa2-24e07bda52af.png)

`?` 데이터는 어떤 데이터와 가장 닮아있을까? 
  
여기서 `?` 데이터는 `세모` 데이터와 가깝이 때문에 `세모` 데이터라고 판단할 수 있다!
  
    여기서 우리가 말하는 '닮아있다' 라는 단어는 데이터 끼리의 'Vector 거리가 서로 가깝다' 고 말할 수 있겠다.   
     
    
</div> 


<br>

#### What's Nearest?

    항상 가까이 있는 데이터가 그 데이터와 가장 닮은 데이터라고 할 수 있을까?

<div align=center>
  
![image](https://user-images.githubusercontent.com/59076451/132226493-8481cf8c-eb91-493b-b440-deb1acdf85ec.png)
  
위 그림에서 볼 수 있 듯,   `?` 데이터는 사실 `동그라미` 데이터와 가~장 가까이 있다.
  
따라서 우리는 `?` 데이터를 `동그라미` 데이터의 범주에 포함시켜주어야할까? 
  
    답은 당연히 NO!
  
반경을 조금 더 넓혀서 보면 점점 뭔가 이상하다는 느낌을 받을 것이다. 
  
우리는 따라서 `주변에 가장 가까운 것이 무엇인가!` 보다는 `주변에 비슷한 데이터가 얼마나 있는가!` 를 보아야 하는 것이다.
  
    실제로 우리가 모델을 사용할 때 고려하게 될 feature 개수는 보통 2개 이상이므로 고차원 데이터를 사용하겠지만, 위 개념을 그대로 적용한다.    
  
    즉, n개의 특성(feature)을 가진 데이터는 n차원의 공간에 점으로 개념화 할 수 있다.
    
</div>  

<br>

위와 같은 방식을 `KNN` 또는 `K-nearest neighborhood algorithm`이라 한다. 

여기서 `K`는 데이터의 개수를 의미한다. 

      주변에 K 개의 비슷한 데이터가 있다면 그 데이터를 K 데이터의 범주로 넣겠다!
      
      예를 들어 K = 1 이라면 '?' 데이터는 동그라미로 분류될 것이다
      하지만 K = 4라면 '?' 데이터는 세모로 분류된다!
      
<br>      

#### what's best for `K` ?

      그렇다면 가장 분류를 잘하기 위해서 'K'를 어떠한 값으로 설정해야할까?
      
<br>      
      
일반적으로 `k`는 `총 데이터의 제곱근`으로 설정한다.

      너무 큰 값은 Outliar를 잘 걸러주는 이점이 있지만, 동시에 분류를 잘 못하는 모습을 보여주기 때문이다.
      
    
<br>    

#### Distance ?

      'Vector 거리가 가까운' 데이터를 알기 위해서 '거리'는 어떻게 알 수 있을까?

<div align=center>  
  
`Vector 간의 거리` 는 `유클라디안 거리`를 사용하면 된다 !
  
![image](https://user-images.githubusercontent.com/59076451/132228440-9dcf0435-cbd6-44de-924d-561ab97b5024.png)
  
</div>  

<br>

#### 실제 Code 구현 시 주의할 점 

KNN과 같은 거리 기반 분류 모델을 구현할 때에는 `데이터의 정규화`가 필수!

      예를 들어 다음과 같이 Feature 간의 분포가 크게 다른 경우 분류에 분명 큰 장애가 생긴다.
      
      매운맛의 범위는 0 ~ 100만 이상까지 매우 범위가 크고, 단맛은 0 ~ 10으로 상대적으로 매우 좁다.
      
      따라서 대부분의 데이터를 '매운맛'으로 분류하게 된다. 

<div align=center>

![image](https://user-images.githubusercontent.com/59076451/132229359-47a2eafb-e2db-48cd-8186-484ae2535416.png)
  

    feature 데이터의 정규화 방법으로는 다음 두 가지가 있다.
  
![image](https://user-images.githubusercontent.com/59076451/132229954-b55b82d4-8925-4fbe-b4fd-18a19447dba8.png)
  

</div>  

