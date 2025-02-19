
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

## **Clustering Output Image**
Below is the **Gaussian Mixture Model (GMM) clustering result**:

![Gaussian Mixture Model Output](https://raw.githubusercontent.com/Ivan123yoo/Assignment-4./main/images/Gaussian%20mixture%20model.png)



## **Image Classification using K-Means Clustering**

In this section, we apply **K-Means clustering** to classify Sentinel-2 satellite imagery, specifically distinguishing between **sea ice and open water**. The goal is to leverage unsupervised learning to detect differences in spectral characteristics of Sentinel-2 bands. This method helps automate image classification, which is useful for environmental monitoring and climate research.

We will:
- Read Sentinel-2 image bands.
- Preprocess the data and apply K-Means clustering.
- Generate a classified image to distinguish different regions.

### **1. Import Necessary Libraries**
The following libraries are used:
- `rasterio` : Reads Sentinel-2 image bands.
- `numpy` : Handles numerical computations.
- `sklearn.cluster.KMeans` : Applies K-Means clustering.
- `matplotlib.pyplot` : Visualizes the classification results.

```python
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define base path to Sentinel-2 imagery
base_path = "/content/drive/MyDrive/AI4EO/W4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/"

# Specify band file paths
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for K-means, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

# Create an empty array for the result, filled with a no-data value (-1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Assign cluster labels to valid data locations
labels_image[valid_data_mask] = labels

# Plot the result
plt.imshow(labels_image, cmap='viridis')
plt.title('K-means clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')

# Save the image
plt.savefig('images/kmeans_classification_output.png')
plt.show()

# Clean up variables
del kmeans, labels, band_data, band_stack, valid_data_mask, X, labels_image
``` 

## **Clustering Output Image**
Below is the K-Means clustering result applied to Sentinel-2 bands:

![K-means Classification Output](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/K%20means%20clustering%20on%20sentinel%202%20bands.png?raw=true)


### **Explanation of the Results**

The displayed image represents the **K-Means clustering classification applied to Sentinel-2 bands**, where the algorithm has grouped pixels into distinct clusters based on their spectral values.

#### **What this shows:**
The image highlights two different clusters (represented by different colors), distinguishing features within the Sentinel-2 satellite imagery. This clustering approach helps in identifying **variations in surface features**, such as differentiating between **sea ice and leads** in the dataset.

#### **Why this is useful:**
- It provides an **unsupervised classification** method to analyze remote sensing imagery.
- Helps in **automating feature detection** in satellite images, reducing manual interpretation.
- Supports further analysis for **environmental monitoring** and **climate studies**.


# **Gaussian Mixture Model (GMM) Clustering on Sentinel-2 Bands**

## **Introduction**
In this section, we apply the **Gaussian Mixture Model (GMM)** clustering algorithm to Sentinel-2 imagery. Unlike K-Means, which assigns data points to the nearest cluster center, **GMM uses probabilistic clustering**, allowing for soft classification of pixels. This method is useful in remote sensing when spectral properties of land, water, and ice overlap.

The objective is to use **Sentinel-2 bands** to classify different surface types in the image.

---

## **1. Import Necessary Libraries**
The following libraries are used:
- `rasterio`: For reading satellite image data.
- `numpy`: For numerical computations and data manipulation.
- `sklearn.mixture.GaussianMixture`: For performing **GMM clustering**.
- `matplotlib.pyplot`: For visualizing the results.

```python
import rasterio
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Paths to the band images
base_path = "/content/drive/MyDrive/AL4EO/W4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/"
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for GMM, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# GMM clustering
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
labels = gmm.predict(X)

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place GMM labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('GMM clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()
```

## **2. Clustering Output Image**
Below is the result of applying **GMM clustering** to the Sentinel-2 bands:

![GMM Clustering Output](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/GMM%20Clustering.png?raw=true)


### **4. Interpretation of Results**

The **GMM clustering output** on Sentinel-2 bands represents the classification of the satellite image into different surface types. The key observations from the result are:

- The **color-coded regions** correspond to different clusters identified by the **Gaussian Mixture Model (GMM)** based on reflectance values from the Sentinel-2 imagery.
- The **dark purple region (left side)** represents areas where the algorithm could not classify data due to missing or low-reflectance values.
- The **yellow and green regions** indicate different terrain features, with the **yellow areas likely corresponding to highly reflective surfaces** such as ice or snow-covered regions.
- The **blue-green areas** might represent water bodies or darker terrain, which have lower reflectance in the selected band.
- The **gradual color transition** shows how GMM captures soft boundaries between different surface types, unlike K-Means which enforces hard clustering.

This result is useful for **remote sensing applications**, helping to distinguish between features such as **ice, open water, and land** using unsupervised learning.




