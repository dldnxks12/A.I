import numpy as np

class KNN():
    def __init__(self, k):
        self.k = k

    def train(self, X, y):
        # load data
        self.X_train = X # data
        self.y_train = y # data에 해당하는 label (group)


    def predict(self, X_test): # Compute Distance and Predict Group 
        distances = self.compute_distance(X_test)  
        return self.predict_labels(distances)

    def compute_distance(self, X_test):
        
        # test point와 모든 data와의 거리를 계산할 것 
        num_test = X_test.shape[0] # test point의 개수 : 5개  (test data : 5 x 2 )
        num_train = self.X_train.shape[0] # data의 개수 : 90개  (data : 90 x 2)

        # distances : 해당 test point와 각 data 간의 거리를 담을 array
        distances = np.zeros( (num_test, num_train) ) # (5 x 90)
 
        # distance between ith test data from jth train data
        # Compute L2 distance
        for i in range(num_test): # Every test point 마다 계산
            for j in range(num_train): # 해당 test point와 모든 data 간의 거리 계산
                distances[i, j] = np.sqrt( np.sum((X_test[i, :] - self.X_train[j, :]) ** 2)) # i번째 test 데이터와 j번 째 데이터 사이 거리 계산 
        return distances

    def predict_labels(self, distances): # test point와 가장 가까운 labels의 그룹

        num_test = distances.shape[0] # test point의 개수 : 5개 
        y_pred = np.zeros(num_test)  # 가장 가까운 label로 분류할 array [0, 0, 0, 0, 0] 

        for i in range(num_test):
            # np.argsort () : 가장 작은 값 순서로 index return
            # distances[i, :] : i 번째 test point에서 다른 data까지의 각각의 거리
            y_indices = np.argsort(distances[i, :])

            # y_indices[:self.k] 작은 순서대로 k개의 index만 사용
            # self.y_train[ y_indices[:self.k] ] k개의 index에 해당하는 y_train 값

            k_closest_classes = self.y_train[ y_indices[:self.k] ].astype(int)
            '''
                    np.bincount : 해당 클래스 당 중복된 값이 몇 개인지, 가장 큰 값을 가진 class return 
                    ex) class = np.array([1,2,2,3,1,4])
                        result = np.bincount(class)
                        result : [0, 2, 3, 1, 1] 
            '''

            y_pred[i] = np.argmax(np.bincount(k_closest_classes))

        return y_pred

if __name__ == "__main__":
    X = np.loadtxt('./data.txt', delimiter=',')
    y = np.loadtxt('./targets.txt')

    KNN = KNN( k = 3 )
    KNN.train(X,y)

    x = np.array([[-2,1]])
    print(x.shape)
    y_pred = KNN.predict(x)
    print(f"ACC {sum(y_pred==y)/y.shape[0]}")



