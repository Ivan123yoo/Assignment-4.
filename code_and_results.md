
first we start of with mounting

from google.colab import drive
drive.mount('/content/drive')

then we install rasterio

pip install rasterio

## K-means Clustering Example

This section of the notebook demonstrates the use of K-means clustering on synthetically generated data to understand the algorithm's application.

### Dependencies Installation

Before running the clustering, ensure that the necessary Python library `netCDF4` is installed. This library is essential for handling scientific data formats:


## **K-means Clustering Example**
This section demonstrates the use of **K-means clustering** to classify randomly generated data into four clusters.

### **1. Import Necessary Libraries**
The following libraries are used:
- `sklearn.cluster.KMeans` : For performing K-means clustering.
- `matplotlib.pyplot` : For plotting the clustering results.
- `numpy` : For generating random sample data.

```python 
# Python code for K-means clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Generate random sample data
X = np.random.rand(100, 2)

# Apply K-means clustering with 4 clusters
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

# Save the plot
plt.savefig('images/kmeans_clustering_output.png')
plt.show()
```
## **Clustering Output Image**
Below is the K-means clustering result:

![K-means Clustering Output](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/kmeans_clustering_output.png?raw=true)

The output image displays the K-means clustering results applied to a randomly generated dataset. The data points are grouped into four distinct clusters, each represented by a different color.

The black circles in the plot indicate the centroids of the clusters, which are the computed centers of each group.
The clustering algorithm effectively groups similar data points together based on their spatial proximity.
The distribution of points suggests how K-means finds structure within the data, even without prior knowledge of any labels.
This visualization demonstrates how unsupervised learning can be used to categorize unlabelled data based on inherent patterns.



## Gaussian Mixture Model (GMM) Clustering Example

This section introduces **Gaussian Mixture Models (GMM)**, which are a probabilistic model for representing normally distributed subpopulations within an overall dataset. Unlike K-means clustering, which assigns each data point to a single cluster, **GMM provides a probability-based clustering approach**, meaning each point has a probability of belonging to multiple clusters. This makes it **more flexible** for identifying clusters with different shapes and variances.

GMM is particularly useful in scenarios where:
- **Soft Clustering is Required**: Instead of a hard assignment like K-means, GMM provides probability estimates for each cluster.
- **Cluster Shape Flexibility**: GMM allows for elliptical and complex cluster structures, unlike K-means which assumes spherical clusters.

### **1. Import Necessary Libraries**
The following libraries are used:
- `sklearn.mixture.GaussianMixture` : For performing Gaussian Mixture Model clustering.
- `matplotlib.pyplot` : For plotting the clustering results.
- `numpy` : For generating random sample data.

```python
# Import required libraries
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Generate random sample data
X = np.random.rand(100, 2)

# Apply Gaussian Mixture Model clustering with 3 components
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Gaussian Mixture Model')

# Save the plot
plt.savefig('images/Gaussian_mixture_model.png')
plt.show()
```


