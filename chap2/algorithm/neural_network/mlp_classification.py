from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)

import mglearn
import matplotlib.pyplot as plt
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=10)\
    .fit(X_train, y_train)

mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()


"""MLP with cancer dataset
"""
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print("Train accuracy: {:.3f}".format(mlp.score(X_train, y_train)))
print("Test accuracy: {:.3f}".format(mlp.score(X_test, y_test)))

# Standardizing datasets
print("Max per features:\n{}".format(cancer.data.max(axis=0)))

mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)

X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train_scaled, y_train)

print("Train accuracy: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Test accuracy: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

mlp = MLPClassifier(max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

print("Train accuracy: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Test accuracy: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=42)
mlp.fit(X_train_scaled, y_train)

print("Train accuracy: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Test accuracy: {:.3f}".format(mlp.score(X_test_scaled, y_test)))