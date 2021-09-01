import numpy as np

class NaiveBayes():
    def __init__(self, X, y): # X는 data matrix (num_examples, num_features), y는 label
        pass

    def fit(self, X, y):  # training part
        pass

    def predict(self, X, y): # prediction part
        pass


    def density_function(self, x, mean, sigma): # Calculate probability from Gaussian Density Function

        '''
        Naive Bayes를 Python으로 구현할 때 주의할 점
           1. 실제 베이즈 정리를 사용할 때에는 평활화 처리가 필요하다. (우도 값 가운데 0이 되는 값이 있기 때문에 이를 방지하기 위해 분모 분자에 모두 작은 값을 추가해줌)
           2. 컴퓨터는 0에 가까운 부동 소수점을 제대로 처리하지 못하므로 우도의 곱셈은 로그 우도의 합으로 처리한다.
        '''
        pass