import mglearn
import matplotlib.pyplot as plt

mglearn.plots.plot_scaling()
plt.show()

"""Scaler
    - StandardScaler
    - RobustScaler
    - MinMaxScaler
    - Normalizer
"""

# Load datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)

print(X_train.shape)
print(X_test.shape)

# Withou Scaler
from sklearn.svm import SVC

svm = SVC(C=100)
svm.fit(X_train, y_train)
print("[{}] Test accuracy: {:.3f}".format("Naive", svm.score(X_test, y_test)))

"""MinMax Scaler
"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
print("[{}] Test accuracy: {:.3f}".format("MinMaxScaler", svm.score(X_test_scaled, y_test)))


"""Standard Scaler
"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
print("[{}] Test accuracy: {:.3f}".format("StandardScaler", svm.score(X_test_scaled, y_test)))


"""Roubst Scaler
"""
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
print("[{}] Test accuracy: {:.3f}".format("RobustScaler", svm.score(X_test_scaled, y_test)))


"""Roubst Scaler
"""
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
print("[{}] Test accuracy: {:.3f}".format("Normalizer", svm.score(X_test_scaled, y_test)))