### 2021 Segmentation Project in NI²L LAB at KW Univ

#### Purpose of this Project  

  - Building Segmentation Model for Ultrasonic-Wave Image Processing for medical purposes

        what to do? 
        
        가공된 데이터를 넣어 질병의 유무를 Classification 해줄 Segmentation 모델을 만들 것
        
---               
                
### To do list

#### 1주차

  - Pytorch remind

        1. Linear Regression   - ok
        2. Logistic Regression - ok
        3. Softmax Regression  - ok
        4. MLP/ANN             - ok
        5. CNN                 - ok

  - Segmentation ?

        1. Segmentation 이란?

  - Code Review

        * code review ( segtrain.py ) - ok
        
        1. argparse - ok 
        2. torch - dataloader - ok
        3. torchvision - ok 

  - 추가 개념 

        1. SOTA Browser 
        
        2. Kaggle 

            Dataset Archive 
          
        3. torchvision - deeplabv3

            Semantic Segmentation model provided by Pytorch
            
              - 현재 UltraSound에서 사용하는 모델 : deeplabv-resnet50 , deeplabv-resnet101         

        4. contigous - ok
        
            데이터 읽어 들이는 순서에 관한 함수
          
        5. permute VS view - ok 
          
            view    : 데이터 읽어들이는 순서 변경 (shape는 사실 변하지 않는다.)
            permute : Dimension Index 순서 변경
              
               permute를 사용할 경우 contigous와 같이 사용하는 경우가 많다.
          
        6. SubsetRandomSampler - ok

            전체 dataset에서 Train dataset과 Test dataset을 각각 부분 집합으로 만들어 dataloader에 넣어주는 방법 

---

#### 2주차

  - Paper & Notation 
    
      1. FCN - ok
      2. SegNet
      3. Deconvolution - ok
      4. deeplabv3
  

  - 기본적인 Segmentation 구현 with Kaggle BUSI Dataset 

        1. 간단한 Segmentation 모델 구현 with BUSI 

  - pytorch deeplabv3 
   
        1. deeplabv3와 BUSI를 이용한 Segmentation 구현 

- 링크 

https://shangom-developer.tistory.com/4<br>
https://github.com/jfzhang95/pytorch-deeplab-xception

---

#### 3주차

- Segmenatation Customization 

    - 사용할 형태로 모델 구현 (필요한 기능이 있다면 추가 구현)  

          1. 입력 데이터 Segmentation
          2. 출력 데이터 Classification
      
    
  
            
        
  
  
  
  
  
