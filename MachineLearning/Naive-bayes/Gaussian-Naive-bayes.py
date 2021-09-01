import numpy as np

'''
Naive Bayes를 Python으로 구현할 때 우도 관련 주의할 점 
   1. 실제 베이즈 정리를 사용할 때에는 평활화 처리가 필요하다. (우도 값 가운데 0이 되는 값이 있기 때문에 이를 방지하기 위해 분모 분자에 모두 작은 값을 추가해줌)
   2. 컴퓨터는 0에 가까운 부동 소수점을 제대로 처리하지 못하므로 우도의 곱셈은 로그 우도의 합으로 처리한다.
'''

class NaiveBayes():
    def __init__(self, X, y): # X는 data matrix (num_examples, num_features), y는 label

        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.eps = 1e-6 # For Numerical Stability (평활화에 사용할 것)

    def train(self, X):  # training part
        '''
            사전확률과 우도를 모두 구하는 과정이지만 여기서는 평균과 분산, 그리고 사전확률만 먼저 구한다.
            우도를 구하는 과정은 아래 density_function에서 진행할 것이다.
        '''
        self.classes_mean = {}     # 해당 범주의 평균
        self.classes_variance = {} # 해당 범주의 분산
        self.classes_prior = {}    # 사전 확률 --- 범주 내의 데이터 / 전체 데이터

        # 모든 클래스에 대해서 평균, 분산, 사전 확률 구하기
        for c in range(self.num_classes):
            X_C = X[y == c] # 해당 label에 속하는 클래스만 True

            self.classes_mean[str(c)] = np.mean(X_C, axis = 0) # dictionary에 key value로써 넣어줌
            self.classes_variance[str(c)] = np.var(X_C, axis = 0)

            self.classes_prior[str(c)] = X_C.shape[0] / self.num_examples

    def predict(self, X): # prediction part
        probs = np.zeros((self.num_examples, self.num_classes)) # 해당 클래스들에 속할 확률들을 담을 array

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = self.likelihood(X, self.classes_mean[str(c)], self.classes_variance[str(c)])

            probs[:, c] = probs_c + np.log(prior) # log를 이용해서 곱셈을 더하기로 바꿈 --- 사전확률x우도 알지?

        return np.argmax(probs, 1) # 가장 높은 확률을 가진 class를 return

    def likelihood(self, x, mean, sigma):
        # Calculate probability from Gaussian Density Function for Likelihood
        # 아래 수식은 잘 이해가 안되지만 차차 이해할 수 있게 하자.
        # 우선 Gaussian Naive Bayes에서는 우도는 정규분포로 바꿔서 구하는데, 그런 과정을 더 나이스하게 바꾼 Form을 사용한 것이다.

        const = -self.num_features/2 * np.log(2*np.pi) - 0.5*np.sum(np.log(sigma+self.eps))
        probs = 0.5*np.sum(np.power(x-mean, 2) / (sigma + self.eps), 1)
        return const - probs



if __name__ == "__main__":

    X = np.loadtxt('./data.txt', delimiter=',')
    y = np.loadtxt('./targets.txt') - 1

    NB = NaiveBayes(X,y)
    NB.train(X) # train
    y_pred = NB.predict(X) # Inference

    print(f"ACC : {sum(y_pred == y) / X.shape[0]}") # ACC