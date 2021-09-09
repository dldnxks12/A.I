#### Python Variable 

`1. Basic Variable`

              - int : 3, 4, .. 
              - float : 3.1 , 4.1, ...
              - boolean : True , False
              
<br>              

`2. Container type Variable`

              - String : "hi" 
              - list : [1,2,3,'kim']
              - tuple : ()
              - dict : {a:1, b:2, ..} - Key : Value 
              - set 

                     *Immutable type (unchangable) : String , tuple, set 
                     *Mutable type   (changable)   : List, dictonary

<br>

`3. Library Support Variable`

              - array (numpy) - contrains only same data type ex) [1,2,3]
              - DataFrame (pandas)
              -  ...

<br>

`4. list vs array `

              [1,2,3] + [4,5,6]
              
              list  = [1,2,3,4,5,6] (집합 연산)
              array = [5,7,9] (Vector 연산)
    
<br>

`5. Dataframe`

              data를 frame형태로 보여주는 data type 
              
              큰 Array에서 데이터를 찾기 굉장히 용이하다. 
              
              
<br>

`6. Call by reference vs Call by Value`

<div align=center>

`list와 dict을 제외한 변수들은 모두 immutable이다.` 
  
![image](https://user-images.githubusercontent.com/59076451/132626837-7fd6257c-fea1-47da-a2fc-a10f9267c8b4.png)
  
 
따라서 a에 새로운 값 7을 할당할 때, `새로운 Object 7을 만들어버리고 a가 이를 참조`하게 만든다.
  
결과적으로 b는 기존 a가 참조하던 3.7을 그대로 참조하고 있고, a는 새로이 만들어진 object 7을 참조한다.  

<br>
  
반면에 `list는 Mutable type이다.  `
  
![image](https://user-images.githubusercontent.com/59076451/132627015-0b84d136-cbe6-400d-a5d1-b671224ce8c3.png)
  
즉, `새로운 Object를 생성하지 않고, 기존의 object를 수정할 뿐`이다.  
  
</div>  

