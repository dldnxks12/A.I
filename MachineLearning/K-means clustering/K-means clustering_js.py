'''
    총 3 단계로 K-means Clustering를 수행할 수 있다.

    1. Initialize : K 개 군집의 Centroid 초기화 --- Random 초기화할 것
    2. Assignment : 모든 데이터를 각 군집에 할당하는 과정
    3. Update     : 할당된 각 군집의 데이터에 대해서 평균을 내어 Centroid로 새로이 할당

    위 1 ~ 3 단계 반복
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class KMeansClustering():
    def __init__(self, X, K):  # Centeroid 초기화 하기
        self.K = K  # Clusters
        self.X = X  # Samples
        self.num_examples = self.X.shape[0]
        self.num_features = self.X.shape[1]

        # Centroid Initialize
        self.centroids = np.zeros((self.K, self.num_features))
        self.Initialize()

    # --- C_k Value 초기화 --- #
    # Random 초기화 방법도 있지만, 걍 Example 중에서 임의로 3개 잡아서 Centroid로 만들자
    # Cluster가 3개니까 Centroid도 3개로 잡아야한다.
    # Centroid : numpy array로 만들어 놓자
    def Initialize(self):
        '''
        centroids :
            shape = (3, 2),
            [[0. 0.]
             [0. 0.]
             [0. 0.]]
        '''
        # Random Centroid Choice
        for i in range(self.K):
            self.centroids[i] = self.X[np.random.choice(range(self.num_examples))]  # X 중에서 1개 골라서 Centeroids에 추가
        print("input Centroids", self.centroids)

    # r_nk Value 할당
    # 해당 data가 어떤 Centroid와 가장 가까운 지 계산해서 r_nk에 1 또는 0 할당
    def Assignment(self):

        # S_pred : 1000 x 3
        S_pred = np.zeros((self.num_examples, self.K))  # 각 Example들의 Cluster들 까지의 거리를 담을 것

        # S : 1000 x 1
        S = np.zeros((self.num_examples, 1))  # 각 Example들이 속하는 Cluster를 담을 것

        for i in range(self.K):
            '''
                1. 데이터와 Centroids와의 거리르 계산
                2. np.argmin으로 최소 찾기 
            '''
            S_pred[:, i] = np.sqrt(
                np.sum((self.X - self.centroids[i, :]) ** 2, axis = 1) + (1e-6)
            )

        S = np.argmin(S_pred, axis = 1) # 각 데이터가 속하는 Cluster
        print("S", S)
        return S

    def Set_Assignment(self):

        S = self.Assignment()
        Clustering = [[] for _ in range(self.K)] # 1 x 3 list 생성
        for i in S:
            if i == 0:
                Clustering[0].append(self.X[i])
            elif i == 1:
                Clustering[1].append(self.X[i])
            elif i == 2:
                Clustering[2].append(self.X[i])

        return Clustering

    def Update(self):
        Clustered_set = self.Set_Assignment()
        for i in range(self.K):
            new_centroid = np.sum(Clustered_set[i], axis = 0) / ( len(Clustered_set[i]) + 1e-6 )
            self.centroids[i] = new_centroid

        return self.centroids

    def loop(self, count):
        for i in range(count):
            # do Assignment - Set_Assignment - Update x count times
            output = self.Update()

            print(output)
        return output

if __name__ == "__main__":

    np.random.seed(1800)  # Fix random Seed
    num_cluster = 3
    # n_samples : sample 개수
    # n_features : sample당 feature 개수 (dimension)
    # centers : 각 Cluster에 대한 centroid
    X, y = make_blobs(n_samples=5, n_features=2, centers=num_cluster)  # return value 중 y는 label이므로 사용하지 않는다.

    KMEAN = KMeansClustering(X = X , K = num_cluster)
    Updated_pos= KMEAN.loop(3)

'''
    plt.figure(figsize = (8,8))
    plt.subplot(1,2,1)
    plt.scatter(X[:, 0], X[:, 1], marker='x', c = y)
    plt.subplot(1,2,2)
    plt.scatter(Updated_pos[:, 0], Updated_pos[:, 1], marker='x')
    plt.show()
'''