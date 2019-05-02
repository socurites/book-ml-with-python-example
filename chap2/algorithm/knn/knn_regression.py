import mglearn
import matplotlib.pyplot as plt

"""Plot
"""
mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()


"""KNeighborsRegressor with wave dataset
"""
# Load datasets
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Fit
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=3)

regressor.fit(X_train, y_train)

# Predict
print("Prediction: {}".format(regressor.predict(X_test)))

# Evaluate
print("R^2: {:.2f}".format(regressor.score(X_test, y_test)))