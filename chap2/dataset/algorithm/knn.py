import mglearn
import matplotlib.pyplot as plt

"""Plot
"""
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()


"""KNeighborsClassifier
"""
# Load datasets
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Fit
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(X_train, y_train)

# Predict
print("Prediction: {}".format(classifier.predict(X_test)))

# Evaluate
print("Accuracy: {:.2f}".format(classifier.score(X_test, y_test)))

# Decision boundary
