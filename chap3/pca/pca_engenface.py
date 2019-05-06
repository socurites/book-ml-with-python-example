"""Extract engenface
with LFW datasets
"""

# Load LFW datasets
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
print(image_shape)


# Plot
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})

for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
plt.show()


# Explore
import numpy as np
counts = np.bincount(people.target)

for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='      ')
    if ( i + 1 ) % 3 == 0:
        print()
print()

# Select 50 images by person
mask = np.zeros(people.target.shape, dtype=bool)

for target in np.unique(people.target):
    # print(mask[np.where(people.target == target)])
    # print(mask[np.where(people.target == target)[0]])
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]


"""Use KNN to classify
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=9)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("1-NN Test accuracy: {:.2f}".format(knn.score(X_test, y_test)))


"""Use PCA with whitening
"""
from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


# Use KNN with PCA
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("1-NN Test accuracy: {:.2f}".format(knn.score(X_test, y_test)))


"""Plot Principal Components(engienface)
"""
fig, axes = plt.subplots(3, 5, figsize=(15,12),
                         subplot_kw={'xticks': (), 'yticks': ()})

for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap='viridis')
    ax.set_title("Principal Component {}".format(i + 1))

plt.show()