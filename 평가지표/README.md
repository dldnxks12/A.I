### 평가지표 

- 평가지표로 사용되는 용어들
    
        1. Accuracy
        2. Precision ok
        3. Recall ok
        4. F1 Score (Harmonic mean of Precision and Recall)
        5. IOU (Intersection over Union) - ok
        6. PR Curve (Precision & Recall Curve)
        7. AP (Average Precision)
        8. mAP (mean Average Precision)
        9. ROC Curve
        10. AUC (Area Under ROC Curve)
        11. Multi-Class ROC
        
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

    모든 검출 결과 중 옳게 검출 한 것의 비율  - Model 관점에서 평가
    
        - TP / (TP + FP) --- 즉, 알고리즘이 True라고 검출해 낸 것들 중 옳게 검출해 낸 비율 ---- 옳은 Positive / (옳은 Positive + 틀린 Positive)

        - ex) 만일 알고리즘이 물체를 5개 검출해냈을 때, 이 중 4개가 잘 검출해낸 것이라면 ? Precision = 4/5 

- Recall

    검출해 내야 하는 것 중에 제대로 검출된 것의 비율 - 정답 관점에서 평가
    
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

    - Precision & Recall Curve
    - AP
    
    
- PR Curve

    PR Curve는 Confidence와 Threshold를 이용하여 물체 인식의 평가지표 신뢰도를 높이는 방법이다.
    
    confidence는 물체를 검출해냈을 때의 자신감으로 생각할 수 있다.
    
    만일 어떤 물체를 검출해내었고, 이 검출 결과에 대한 Confidence가 70%라고 하자.
    
    이 때, Threshold를 50%라고 하면 위의 검출을 인정한다. 만일 50% 이하면 검출 결과를 무시한다.
    
Ex) 15개의 얼굴이 존재하는 어떤 데이터셋 에서 한 얼굴 알고리즘에 의해서 총 10개의 얼굴을 검출해냈다고 해보자.    
    
![555](https://user-images.githubusercontent.com/59076451/128005120-a7f394d9-ace9-4cc2-8f1a-1fc5e88bb7aa.PNG)

    - A 검출 결과에 대해서는 알고리즘이 57%의 자신감을 갖는다. 또한 실제로 TP이다.
    - J 검출 결과에 대해서는 알고리즘이 81%의 자신감을 가졌고, 실제로 TP이다.
    
    위 결과에 대해서 
    
        Precision = 7/10 = 0.7
        Recall    = 7/15 = 0.47
    
    위 결과는 threshold에 따른 filtering을 거치지 않은 Precision과 Recall 값이다.
    
![666](https://user-images.githubusercontent.com/59076451/128005123-711714b8-9d34-4925-9a13-6341235e1183.PNG)

    - Confidence에 따라 정렬한 결과이다. 
    
    위 결과에 대해서 만일 아주 엄격하게 Threshold를 정하여 95% 이상에 대해서만 인정한다면 
    
        Precision = 1/1 = 1
        Recall    = 1/15 = 0.067
        
    만일 Threshold를 91%로 한다면 
    
        Precision = 2/2 = 1
        Recall    = 2/15 = 0.13
        
        ...
        
    이런식으로 Threshold 를 점점 낮춰가며 계산해보면 다음과 같은 Precision과 Recall에 대한 결과 표가 만들어진다.

![888](https://user-images.githubusercontent.com/59076451/128005128-15009fe8-4007-4854-9ec2-f782e4223a84.PNG)

    위 결과를 Graph로 만들어낸 것이 아래의 PR Curve이다.
    
![999](https://user-images.githubusercontent.com/59076451/128005129-d29f7d78-363a-44a9-bd3b-82cbeb70decf.PNG)

    
       

    
    


