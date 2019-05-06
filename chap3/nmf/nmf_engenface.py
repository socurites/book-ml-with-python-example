# Load LFW datasets
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
print(image_shape)


# Select 50 images by person
import numpy as np
mask = np.zeros(people.target.shape, dtype=bool)

for target in np.unique(people.target):
    # print(mask[np.where(people.target == target)])
    # print(mask[np.where(people.target == target)[0]])
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=9)


"""NMF
"""
from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)

X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=9)

# Use KNN with NMF
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_nmf, y_train)

print("1-NN Test accuracy: {:.2f}".format(knn.score(X_test_nmf, y_test)))



"""Plot Components
"""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 5, figsize=(15,12),
                         subplot_kw={'xticks': (), 'yticks': ()})

for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap='viridis')
    ax.set_title("Component {}".format(i + 1))

plt.show()