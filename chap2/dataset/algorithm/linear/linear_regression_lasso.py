"""Lasso with Boston dataset
"""

import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import numpy as np

# Load Dataset
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lasso_regressor = Lasso()
lasso_regression = lasso_regressor.fit(X_train, y_train)
print("Train R^2: {:.2f}".format(lasso_regression.score(X_train, y_train)))
print("Test R^2: {:.2f}".format(lasso_regression.score(X_test, y_test)))
print("# of used features: {}".format(np.sum(lasso_regression.coef_ != 0)))


lasso_regressor = Lasso(alpha=0.01, max_iter=100000)
lasso_regression = lasso_regressor.fit(X_train, y_train)
print("Train R^2: {:.2f}".format(lasso_regression.score(X_train, y_train)))
print("Test R^2: {:.2f}".format(lasso_regression.score(X_test, y_test)))
print("# of used features: {}".format(np.sum(lasso_regression.coef_ != 0)))

alphas = [1, 0.1, 0.01, 0.001]
labels = ['^', 's', 'v', 'o']
len_coef = 0
for alpha, label in zip(alphas, labels):
    lasso_regressor = Lasso(alpha=alpha, max_iter=100000)
    lasso_regression = lasso_regressor.fit(X_train, y_train)
    print("Alpha: {}".format(alpha))
    print("Train R^2: {:.2f}".format(lasso_regression.score(X_train, y_train)))
    print("Test R^2: {:.2f}".format(lasso_regression.score(X_test, y_test)))

    plt.plot(lasso_regression.coef_, label, label="Alpha: {}".format(alpha))
    len_coef = len(lasso_regression.coef_)

plt.xlabel('Coef')
plt.ylabel('Ceof Value')
plt.hlines(0, 0, len_coef)
plt.ylim(-25, 25)
plt.legend()
plt.show()