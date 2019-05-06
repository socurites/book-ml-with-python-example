from sklearn.datasets import load_digits
digits = load_digits()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 5, figsize=(10, 5),
                         subplot_kw={'xticks': (), 'yticks': ()})

for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)

plt.show()


"""Use t-sne"""
from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)

for i in range(len(digits.data)):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             # color=colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})

plt.xlabel('t-SNE Feature 0')
plt.xlabel('t-SNE Feature 1')
plt.show()