"""Ridge with Boston dataset
"""

import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# Load Dataset
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


ridge_regressor = Ridge()
ridge_regression = ridge_regressor.fit(X_train, y_train)
print("Train R^2: {:.2f}".format(ridge_regression.score(X_train, y_train)))
print("Test R^2: {:.2f}".format(ridge_regression.score(X_test, y_test)))

alphas = [10, 5, 1, 0.1]
labels = ['^', 's', 'v', 'o']
len_coef = 0
for alpha, label in zip(alphas, labels):
    ridge_regressor = Ridge(alpha=alpha)
    ridge_regression = ridge_regressor.fit(X_train, y_train)
    print("Alpha: {}".format(alpha))
    print("Train R^2: {:.2f}".format(ridge_regression.score(X_train, y_train)))
    print("Test R^2: {:.2f}".format(ridge_regression.score(X_test, y_test)))

    plt.plot(ridge_regression.coef_, label, label="Alpha: {}".format(alpha))
    len_coef = len(ridge_regression.coef_)

plt.xlabel('Coef')
plt.ylabel('Ceof Value')
plt.hlines(0, 0, len_coef)
plt.ylim(-25, 25)
plt.legend()
plt.show()


# Learning curve by size of training set
mglearn.plots.plot_ridge_n_samples()
plt.show()