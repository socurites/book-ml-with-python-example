"""Linear Regression
OLS: Ordinary Least Squares
"""

import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""LinearRegressiopn with wave dataset
"""
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regression = regressor.fit(X_train, y_train)

print("coef_: {}".format(regression.coef_))
print("intercept_: {}".format(regression.intercept_))

print("Train R^2: {:.2f}".format(regression.score(X_train, y_train)))
print("Test R^2: {:.2f}".format(regression.score(X_test, y_test)))


"""LinearRegression with Boston dataset
"""
X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regressor = LinearRegression()

regression = regressor.fit(X_train, y_train)
print("Train R^2: {:.2f}".format(regression.score(X_train, y_train)))
print("Test R^2: {:.2f}".format(regression.score(X_test, y_test)))