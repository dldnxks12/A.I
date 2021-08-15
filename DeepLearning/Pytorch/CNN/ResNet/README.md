### ResNet - deep residual learning for image recognition

신경망은 depth를 늘릴 수록 학습에 관여하는 파라미터가 기하급수적으로 증가한다

    즉, Computing power와 Overfitting 면에서의 좋지 않은 성능을 보여준다
    
ResNet은 Skip Connection 기법을 이용해서 depth 늘림에도 Overfitting이 발생하거나 Computing Power가 커지는 부분을 어느정도 줄여주도록 한다.


#### Skip Connection

Gradient Vanishing/Exploding 문제를 해결하기 위한 기법

    입력 x를 몇 layer 이후의 출력 값에 더해주는 방법 

![image](https://user-images.githubusercontent.com/59076451/129482581-13c74fcc-2901-4fa0-aa71-de1572cf4b93.png)


#### Bottle neck

GoogLeNet과 NIN에서 사용했던 방법으로, 학습 성능은 유지하면서 관여하는 매개변수 수를 줄이는 방법 



