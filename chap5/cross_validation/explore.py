from sklearn.datasets import load_iris

iris = load_iris()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="lbfgs", multi_class='auto', max_iter=1000)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr, iris.data, iris.target, cv=3)

print("Cross-validation scores: {}".format(scores))
print("Avg Cross-validation scores: {:.2f}".format(scores.mean()))