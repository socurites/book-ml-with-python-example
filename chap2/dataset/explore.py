import mglearn
import matplotlib.pyplot as plt

"""Factitious forge dataset for binary classification
"""
X, y = mglearn.datasets.make_forge()
print("X.shape: {}".format(X.shape))

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=1)
plt.xlabel("first feature")
plt.ylabel("second feature")
plt.show()


"""Factitious wave dataset for regression
"""
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()


"""Cancer dataset: Wisconsin Breast Cancer
"""
from sklearn.datasets import load_breast_cancer
import numpy as np
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("cancer.data.shape: {}".format(cancer.data.shape))
print("cancer.target.shape: {}".format(cancer.target.shape))

print("cancer.feature_names: \n{}".format(cancer.feature_names))
print("cancer.target_names: \n{}".format(cancer.target_names))


print("Number of samples per class: \n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))


"""Boston dataset: Boston Housing
"""
from sklearn.datasets import load_boston
boston = load_boston()
print("boston.keys(): \n{}".format(boston.keys()))
print("boston.data.shape: {}".format(boston.data.shape))
print("boston.target.shape: {}".format(boston.target.shape))

print("boston.feature_names: \n{}".format(boston.feature_names))


# Extends dataset
import mglearn
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))