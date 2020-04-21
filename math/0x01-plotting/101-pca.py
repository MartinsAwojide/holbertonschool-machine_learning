#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Create a figure
# Axes3d enables 3d plotting.
fig = plt.figure()
ax = Axes3D(fig)

x = pca_data[:, 0]
y = pca_data[:, 1]
z = pca_data[:, 2]

ax.scatter(x, y, z, c=labels, cmap=plt.cm.plasma)
# cmpa: if c is an array => labels is an array of 0's, 1's & 2's
# for each axis 0:x, 1:y, 2:z

ax.set_xlabel("U1")
ax.set_ylabel("U2")
ax.set_zlabel("U3")
fig.suptitle('PCA of Iris Dataset')
plt.show()
