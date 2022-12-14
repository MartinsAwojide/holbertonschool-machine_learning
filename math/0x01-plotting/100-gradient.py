#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

fig, ax = plt.subplots()
im = ax.scatter(x, y, c=z, s=40)
# c: color, s: scalar

# add the colorbar
fig.colorbar(im, ax=ax, label="elevation (m)")

plt.xlabel('x coordinate (m)')
plt.ylabel('y coordinate (m)')
plt.suptitle('Mountain Elevation')
plt.show()
