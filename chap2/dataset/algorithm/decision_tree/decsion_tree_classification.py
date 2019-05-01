import mglearn
from sklearn.tree import DecisionTreeClassifier

# Load datasets
from sklearn.datasets import load_breast_cancer
import numpy as np
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("cancer.data.shape: {}".format(cancer.data.shape))
print("cancer.target.shape: {}".format(cancer.target.shape))

print("cancer.feature_names: \n{}".format(cancer.feature_names))
print("cancer.target_names: \n{}".format(cancer.target_names))


print("Number of samples per class: \n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print("Train accuracy: {:.3f}".format(tree.score(X_train, y_train)))
print("Test accuracy: {:.3f}".format(tree.score(X_test, y_test)))

# Pruning
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("Train accuracy: {:.3f}".format(tree.score(X_train, y_train)))
print("Test accuracy: {:.3f}".format(tree.score(X_test, y_test)))

# Graphviz
from sklearn.tree import export_graphviz
out_file = "/home/socurites/Download/tree.dot"
export_graphviz(tree, out_file=out_file, class_names=["Negative", "Positive"],
                feature_names=cancer.feature_names,
                impurity=False, filled=True)

# Feature Importance
print("Feature importance:\n{}".format(tree.feature_importances_))

import matplotlib.pyplot as plt
def plot_feature_importance(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importance")
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)
    plt.show()

plot_feature_importance(tree)
