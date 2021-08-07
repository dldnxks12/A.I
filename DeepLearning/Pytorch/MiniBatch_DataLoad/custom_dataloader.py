'''

https://wikidocs.net/57165

파이토치에서 데이터셋을 좀 더 쉽게 다룰 수 있도록 유용한 도구로서 torch.utils.data.Dataset과 torch.utils.data.DataLoader를 제공
기본 사용 법은 Dataset을 정의하고 이를 DataLoader에 전달하는 것이었다.
하지만 이 torch.utils.data.Dataset을 상속받아서 직접 데이터셋을 커스텀하는 경우가 많다.

다음은 이 Dataset을 상속받아 다음 method 들을 Override 해서 데이터셋을 만드는 방법이다.


'''

