### Basic Notion

<br>          

<br>        

#### `Basic Notion`

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


`7. '==' and 'is'`

<div align=center>


`==` : Object의 값이 같은 가?

`is` : Object가 같은 가?

![image](https://user-images.githubusercontent.com/59076451/132628145-82f617bf-a32b-4cdd-8a80-19aeb9e73d5a.png)

</div>


<br>          

<br>          

#### `Function`

- `__main__`

`if __name__ == "__main__"` 

    main 함수는 해당 .py에서 실행할 경우 실행되며, 해당 .py 파일을 Import 해서 사용할 경우 실행되지 않는다.

<br>          

<br>          

#### `Class`

    class: 붕어빵을 만들기 위한 기본 틀과 같은 개념 (혹은 운전면허등을 만들기 위한 기본 틀)
    instance: 이 틀로 만들어지는 각각의 붕어빵들 (각각의 개인 정보를 넣어 여러 면허증을 만들어 냄)
    self 변수: class 와 instance 를 연결해주는 역할
    _init_() 함수 : class 에서 사용할 변수를 정의, 매개변수의 첫번째는 항상 self
    _str_() 함수 : instance 자체를 print 로 출력하면 나오는 값

- `Class 상속`

`Class 상속에서 Super의 의미는 상속한 Class의 __init__ 내의 선언을 따를 것인지의 여부이다.`

```python
# advanced issue (inheritance)
# super(): parent class

class Parent:
    def __init__(self, p1, p2):
        '''super()를 사용하지 않으면 overriding 됩니다.'''
        self.p1 = p1
        self.p2 = p2
    p3 = "Dummy"
        
class Child_1(Parent):
    def __init__(self, c1, **kwargs): # Super의 미사용으로 __init__()을 Overriding 함         
        self.c1 = c1
        self.c2 = "This is Child's c2"
        self.c3 = "This is Child's c3"

class Child_2(Parent):
    def __init__(self, c1, **kwargs): # Super의 사용으로 __init__()을 그대로 받아 
        super(Child_2, self).__init__(**kwargs)
        self.c1 = c1
        self.c2 = "This is Child's c2"
        self.c3 = "This is Child's c3"
        # print(kwargs)
```
