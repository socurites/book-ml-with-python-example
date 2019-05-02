import mglearn
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(15, 2, figsize=(10, 20))

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())

ax[0].set_xlabel("Feature size")
ax[0].set_ylabel("Frequency")
ax[0].legend(["Malignant", "Benign"], loc="best")
fig.tight_layout()
plt.show()


"""PCA used
"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit(cancer.data).transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit(X_scaled).transform(X_scaled)

plt.figure(figsize=(8,8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(["Malignant", "Benign"], loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("1st Principal component")
plt.ylabel("2nd Principal component")
plt.show()