import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from fitfunction import fit_gmm

# Generate some sample data
X, y = make_blobs(n_samples=1000, centers=3, random_state=42)

# Fit the GMM to the data
gmm = fit_gmm(X, n_components=3)

# Make predictions for new data points
x_new = np.array([[0, 0], [5, 5], [10, 10]])
probas = gmm.predict_proba(x_new)

# Plot the results
colors = ['r', 'g', 'b']
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], color=colors[y[i]], alpha=0.5)
plt.scatter(x_new[:, 0], x_new[:, 1], s=100, marker='s', color=colors)
plt.show()

print(probas)
