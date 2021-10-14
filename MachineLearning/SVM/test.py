from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

classifier = SVC(kernel = 'linear')

training_points = np.array([[1, 2], [1, 5], [2, 2], [7, 5], [9, 4], [8, 2]])
labels = np.array([1, 1, 1, 0, 0, 0])

print(training_points.shape)
x1 = training_points[:, 0]
x2 = training_points[:, 1]
print(x1.shape)
classifier.fit(training_points, labels)

vectors = classifier.support_vectors_
print(vectors)

print(classifier.predict([[3,5]]))


'''
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.scatter(x1, x2, c=labels)

plt.subplot(1,2,2)
plt.scatter(vectors[:,0], vectors[:,1], c='r')
plt.show()
'''


