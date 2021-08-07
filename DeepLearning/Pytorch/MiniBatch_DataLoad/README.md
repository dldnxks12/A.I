#### Mini Batch , Batch Size and Iteration 

- Mini Batch and Batch Size

  우리가 다룰 데이터가 만약 방대한 양이라면?

  경사 하강법을 전체 데이터에 대해서 몇 번이고 반복해서 학습하는 과정이 메모리에 굉장한 부담을 주게 된다.

  그렇게 때문에 전체 데이터를 나누어서 해당 단위로 학습하는 개념이 나온 것! 이 단위를 미니 배치라 한다. (Mini Batch)

  ![111](https://user-images.githubusercontent.com/59076451/128611769-b0993a6d-443e-4931-ac6e-2b487b2f1fd2.PNG)

  Mini batch 학습 과정

      미니 배치 학습을 하게 되면, 해당 양만큼만 가져가서 그에 대한 cost를 계산하고, 경사하강법을 수행한다.

      그 다음 미니 배치를 또 가져가서 같은 과정을 반복한다.

      이렇게 모든 미니 배치들을 가져가서 전체 데이터에 대한 학습을 1회 끝마치면 1 Epoch이 종료된다. 

      이 미니 배치의 크기를 Batch size라고 한다. 
    
- Iteration
     
  Epoch, Batch 그리고 Iteration의 관계는 다음과 같다.

      방대한 데이터를 Batch Size로 나누어 미니 배치를 만든다고 했다. 이 미니 배치 만큼 떼어가서 학습을 진행하고, 

      조금씩 떼어가서 학습한 데이터가 전체 데이터가 되면 1 Epoch을 돌았다고 한다.

      이 때, 미니 배치를 몇 번 돌아야 1 Epoch일까? 

      여기서 '몇 번'에 해당하는 횟수가 Iteration이다. 
    
![image](https://user-images.githubusercontent.com/59076451/128611856-dd8c1d81-0da6-4505-a50c-24d2460ed39d.png)    

#### Data Load 

파이토치에서는 데이터를 좀 더 쉽게 다룰 수 있게 유용한 도구로서 Dataset , DataLoader를 제공한다.

위 두 가지 API로 미니 배치 학습, 데이터 섞기(Shuffle), 병렬 처리 까지 간단히 수행할 수 있다.

기본적인 사용 방법은 Dataset을 정의하고, 이를 DataLoader에 전달하는 것이다. 

    - 실습 코드 참고
