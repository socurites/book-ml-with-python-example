import mglearn
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_wave(n_samples=100)
print("X.shape: {}, y.shape: {}".format(X.shape, y.shape))

import numpy as np
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
print("line.shape: {}".format(line.shape))


# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, dt_regressor.predict(line), label="Decision Tree")

# Linear Regression
from sklearn.linear_model import LinearRegression
l_regressor = LinearRegression().fit(X, y)
plt.plot(line, l_regressor.predict(line), '--', label="Linear Regression")

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input Features")
plt.legend(loc='best')
plt.show()


"""Feature Transform: Binning
"""
bins = np.linspace(-3, 3, 11)
print("Bin: {}".format(bins))

which_bin = np.digitize(X, bins=bins)


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)


line_binned = encoder.transform(np.digitize(line, bins=bins))

dt_regressor = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, dt_regressor.predict(line_binned), label="Binned Decision Tree")

# Linear Regression
from sklearn.linear_model import LinearRegression
l_regressor = LinearRegression().fit(X_binned, y)
plt.plot(line, l_regressor.predict(line_binned), '--', label="Binned Linear Regression")

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input Features")
plt.legend(loc='best')
plt.show()


"""Feature Transform: Interaction & Polynomial
"""


"""Feature Transform: Nonlinear
"""
rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

print("X_org.shape: {}".format(X_org.shape))
print("X_org[0].shape: {}".format(X_org[0].shape))

X = rnd.poisson(10 * np.exp(X_org))
print("X.shape: {}".format(X.shape))
print("X[0].shape: {}".format(X[0].shape))

y = np.dot(X_org, w)
print("y.shape: {}".format(y.shape))
print(X[:10])
print(X[:10, 0])