import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from fitfunction import fit_gmm

# Generate some sample data
X, y = make_blobs(n_samples=1000, centers=5, random_state=42)

# Plot the original signal distribution
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], color='gray', alpha=0.5)
plt.title('Original Signal Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Fit the GMM to the data
gmm = fit_gmm(X, n_components=5)

# Plot the segmented signal distribution
colors = ['r', 'g', 'b', 'c', 'm']
plt.figure(figsize=(8, 6))
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], color=colors[y[i]], alpha=0.5)
plt.title('Segmented Signal Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
