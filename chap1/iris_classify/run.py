"""I RIS Classification Example
"""

""" Load Dataset
"""
from sklearn.datasets import load_iris

iris_dataset = load_iris()
print(iris_dataset.keys())


""" Split Dataset
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


"""Explore Dataset
"""