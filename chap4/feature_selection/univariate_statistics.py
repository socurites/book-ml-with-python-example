from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
print("cnacer.data.shape: {}".format(cancer.data.shape))


import numpy as np
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
print("noise.shape: {}".format(noise.shape))

X_w_noise = np.hstack([cancer.data, noise])
print("X_w_noise.shape: {}".format(X_w_noise.shape))

X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)

select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)

X_train_selected = select.transform(X_train)

print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))

# Print selected features
mask = select.get_support()
print(mask.shape)

print(mask.reshape(1, -1).shape)

import matplotlib.pyplot as plt
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Feature Number")
plt.show()
