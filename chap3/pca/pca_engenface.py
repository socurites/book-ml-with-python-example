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
