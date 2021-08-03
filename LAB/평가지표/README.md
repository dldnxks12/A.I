### 평가지표 

- 평가지표로 사용되는 용어들
    
        1. Accuracy
        2. Precision ok
        3. Recall ok
        4. F1 Score (Harmonic mean of Precision and Recall)
        5. IOU (Intersection over Union)
        6. AP (Average Precision)
        7. ROC Curve
        8. AUC (Area Under ROC Curve)
        9. Multi-Class ROC
        10. PR Curve
        ...

Object Detecting의 평가는 보통 PR curve 와 AP 로 평가한다.

    - 이에 대한 이해를 위해 먼저 Precision과 Recall에 대해 알아야함
    
![11111](https://user-images.githubusercontent.com/59076451/128000690-15e802cc-034a-42ee-9ddc-b14bf44aded2.PNG)

        TP(True Positive)  : 실제 Positive - 검출 Positive
            - 옳은 검출 (Good)
            
        TN(True Negative)  : 실제 Negative - 검출 Negative
            - 검출하면 안되는 것을 검출 안함 (Good)
            
        FP(False Positive) : 실제 Negative - 검출 Positive
            - 검출하면 안되는 것을 검출함 (Bad)
        
        FN(False Negative) : 실제 Positive - 검출 Negative
            - 검출해야 되는 것을 검출 안함 (Bad)

- Precision

    모든 검출 결과 중 옳게 검출 한 것의 비율 
    
        - TP / (TP + FP) --- 즉, 알고리즘이 True라고 검출해 낸 것들 중 옳게 검출해 낸 비율 ---- 옳은 Positive / (옳은 Positive + 틀린 Positive)

        - ex) 만일 알고리즘이 물체를 5개 검출해냈을 때, 이 중 4개가 잘 검출해낸 것이라면 ? Precision = 4/5 

- Recall

    검출해 내야 하는 것 중에 제대로 검출된 것의 비율 
    
        - TP / (TP + FN) --- 알고리즘이 마땅히 검출해야하는 것들 중에서 제대로 검출해낸 것의 비율 
        
        - (검출해야하는 것 검출) / {(검출해야하는 것 검출) + (검출해야하는 것을 검출 안함)}
        
- IOU (Intersection of Union)

    IOU는 '옳은 검출'(TP)와 '틀린 검출'(FP) 를 구분하는 기준 

![333](https://user-images.githubusercontent.com/59076451/128002292-a766be23-a7e7-4d5b-9e96-333c38b7e7cd.PNG)

Red box는 검출되야할 물체를 감싸고 있고, Green Box는 예측된 Boundary

    이 때 Green box가 감싸고 있는 물체가 잘 검출된 것인지 어떤식으로 결정하는 것이 좋은가?
    
    IOU는 예측된 박스 실제 라벨 박스 간의 중첩되는 면적을 합집합의 면적으로 나누어준다. 
 
        IOU 값이 0.5 이상이면 제대로 검출(TP)라고 판단.
        
        반면 0.5 미만이면 FP 라고 판단

#### Precision and Recall Trade off

Precision과 Recall은 평가 관점이 다르다. 또한 Precision이 높으면 Recall이 낮고, Precision이 낮으면 Recall이 낮은 trade off 관계를 가진다.

따라서 알고리즘의 성능을 적절히 평가하기 위해 두 평가 지표를 적절히 사용해야한다. 

    - AP
    - Precision & Recall Curve
   
       

    
    


