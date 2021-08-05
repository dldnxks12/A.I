
### Pytorch에서 제공하는 DataLoad 사용법

  Dataloader? 
  
  data 개수가 많을 때, 모든 data 전체를 훑으며 gradient를 학습시켜나가기 힘들다. 
  
  이를 위해 batch를 나누어 학습을 진행하는데, 이에 편의성을 위해 pytorch에서 제공하는 Dataloader를 사용한다.
 
  - Epoch
  - Batch
  - Iteration
  
        - 1000 개의 data가 있을 때, batch size = 500이라면? --- 1 epoch을 돌기 위해 2 번의 iteration이 필요
          
