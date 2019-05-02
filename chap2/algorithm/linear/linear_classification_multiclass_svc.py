import mglearn
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

# Load datasets
from sklearn.datasets import make_blobs
import numpy as np

X, y = make_blobs(random_state=42)
print("X.shape: {}".format(X.shape))
print("y.shape: {}".format(y.shape))

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feture 0")
plt.xlabel("Feture 1")
plt.legend(["Class 0", "Class 1", "Class 2"])
plt.show()

linear_svm = LinearSVC().fit(X, y)
print("Size of weights array: {}".format(linear_svm.coef_.shape))
print("Size of biases array: {}".format(linear_svm.intercept_.shape))

