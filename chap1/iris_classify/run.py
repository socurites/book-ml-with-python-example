"""I RIS Classification Example
"""

""" Load Dataset
"""
from sklearn.datasets import load_iris

iris_dataset = load_iris()
print(iris_dataset.keys())
print(iris_dataset.feature_names)


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
import pandas as pd
import matplotlib.pyplot as plt
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(frame=iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.show()


"""KNN
"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


"""Predict
"""
import numpy as np
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted Label: {}".format(iris_dataset['target_names'][prediction]))


"""Evaluation
"""
y_pred = knn.predict(X_test)
print("Accuracy: {:.2f}".format(np.mean(y_pred == y_test)))
print("Accuracy: {:.2f}".format(knn.score(X_test, y_test)))
