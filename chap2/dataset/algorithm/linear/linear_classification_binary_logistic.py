import mglearn
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load datasets
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("cancer.data.shape: {}".format(cancer.data.shape))
print("cancer.target.shape: {}".format(cancer.target.shape))

print("cancer.feature_names: \n{}".format(cancer.feature_names))
print("cancer.target_names: \n{}".format(cancer.target_names))

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

logit_regression = LogisticRegression().fit(X_train, y_train)
print("Train accuracy: {:.3f}".format(logit_regression.score(X_train, y_train)))
print("Test accuracy: {:.3f}".format(logit_regression.score(X_test, y_test)))

C_vals = [100, 1, 0.1, 0.01]
labels = ['^', 's', 'v', 'o']
for C_val, label in zip(C_vals ,labels):
    logit_regression = LogisticRegression(C=C_val).fit(X_train, y_train)
    print("C_val: {}".format(C_val))
    print("Train accuracy: {:.3f}".format(logit_regression.score(X_train, y_train)))
    print("Test accuracy: {:.3f}".format(logit_regression.score(X_test, y_test)))