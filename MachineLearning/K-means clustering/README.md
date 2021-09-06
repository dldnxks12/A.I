## `K-means Clustering`

      K-평균 군집화 알고리즘은 일정량의 데이터를 받아 이를 그룹으로 묶어주는 비지도 `학습 알고리즘`이다. 

<br>

#### K-means Clustering Algorithm?

      이 알고리즘은 아래 그림과 같이 label이 없는 데이터를 입력받아, 각 데이터에 label을 할당함으로써 군집화를 수행한다.
      
<div align=center>      
  
![image](https://user-images.githubusercontent.com/59076451/132254519-eee65c33-d57e-484b-82aa-254b8786b4ce.png)
  
</div>  

    K-means clustering은 Vector 형태로 표현된 N개의 데이터 X = {x1, x2, ... , xN}에 대해서 
    데이터가 속한 군집(Cluster)의 중심과 데이터 간의 거리의 차이가 최소가 되도록 데이터들을 K개의 군집 S = {s1, s2, ... , sK}에 할당한다.
    
    여기서 군집의 개수 K는 일반적으로 데이터를 분석하고자 하는 사람이 직접 설정한다.
    
    
<br>

#### How it works?

    군집의 개수 K를 설정하였다면, 다음과 같은 최적화 문제로 K-means Clustring을 수행한다.
    
<div align=center>    
  
![image](https://user-images.githubusercontent.com/59076451/132254762-a64d1483-f805-4f25-87d2-961dce99abd1.png)  
  
    r_nk : n번째 데이터가 K 번째 군집에 속하면 1, 그렇지 않으면 0
    c_k  : K 번째 군집의 중심  
  
 <br>
  
위의 수식을 잘 살펴보면 다음과 같다.
  
k번 째 군집에 속하는 n번째 데이터 x에 대해서 (r_nk = 1) 해당 군집의 중심 c_k와의 거리를 구한다.
  
    착각하지 않아야 할 것은 r_nk는 하나의 군집에 대해서만 1의 값을 갖지는 않는다는 것이다.
  
만일 어떠한 데이터 x가 1번 째 군집과 2번 째 군집에 동시에 속한다면?
  
        해당하는 군집의 중심과의 거리를 계산한 후 argmin 함수에 따라 가장 작은 군집으로 포함된다.
  
  
K-means Clustering을 수행한다는 것은 주어진 데이터 X에 대해 r_nk과 c_k 값을 설정하는 것과 같다.  
  
</div>  

<br>

#### How to implement alogorithm?

    위의 식의 형태로 표현된 K-means clustering 문제를 풀기 위한 방법으로는 Lloyd 알고리즘이 주로 사용된다.
    
<div align=center>    
  
![image](https://user-images.githubusercontent.com/59076451/132255196-d63cabfd-171e-45b9-8c67-8245dcaa6613.png)
  
위 알고리즘은 크게 3단계로 구성된다.

    1. Initialization
    2. Assignment
    3. Update  

</div>  

<br>

- Initialization Step 

      초기 C_k의 값을 설정한다. 여러 가지 방법이 있지만 가장 기본은 Random으로 초기화하는 것이다.

<br>

- Assignment Step

      모든 data에 대해서 가장 가까운 Cluster의 중심 C_k를 선택한다.

      데이터와 Cluster와의 거리는 해당 데이터와 Cluster의 중심 C_k의 벡터 거리로 계산한다.

      이 과정을 통해 r_nk 변수에 값을 설정한다.

<div align=center>
  
![image](https://user-images.githubusercontent.com/59076451/132255443-d8174d69-7480-4c5b-bcbd-cab15f1fab73.png)
  
</div>

<br>

- Update Step

        모든 data에 대해서 가장 가까운 군집(cluster)가 선택되면 이를 바탕으로 C_k를 수정한다.
    
        이 단계에서 C_k는 k번째 Cluster S_k에 할당된 모든 데이터들의 평균으로 갱신한다!
                       
<br>    


`K-means Clustering`은 주로 1. i 번 만큼 반복하거나, Cluster의 Center의 변화가 없을 때까지 assignment와 Update를 반복한다.

    주의할 것은 Lloyd 알고리즘은 식 (1)을 전역 최적화가 아닌 지역 최적화 시키는 알고리즘이기 때문에 C_k의 초기화가 매우 중요하다.
    
        C_k의 초기화를 어떻게 하느냐에 따라 매번 결과가 달라진다. 
        
        K 값 또한 마찬가지이다!
        
        
<br>        

[예시 링크](https://untitledtblog.tistory.com/132)







