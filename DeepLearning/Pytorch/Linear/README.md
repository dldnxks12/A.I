#### optimizer.zero_grad()가 필요한 이유

    파이토치는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적시키는 특징이 있다.

      - zero_grad 실습 참고

#### Autograd 

    required_grad = True , backward() ...



    경사 하강법 코드에서 required_grad = True 나 backward() 함수를 보게 된다.

    이는 pytorch에서 제공하는 자동 미분 기능을 수행하는 것이다.

      - autograd 실습 참고 

autograd의 Variable에 대해서도 알아두는 것이 좋을 것 같다. [autograd]https://data-panic.tistory.com/8    
    
   


