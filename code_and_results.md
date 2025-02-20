# Unsupervised Learning for Sea Ice and Lead Classification

This project explores the use of unsupervised learning methods to classify sea ice and leads using Sentinel-2 optical data and Sentinel-3 altimetry data. Instead of relying on labeled datasets, we use clustering techniques to uncover patterns in the data, allowing us to group similar observations.

### Objectives
- Classify sea ice and lead using unsupervised learning methods.
- Apply K-Means and Gaussian Mixture Model (GMM) clustering to satellite-derived datasets.
- Analyze and interpret results using confusion matrices, classification reports, and waveform visualizations.

### Approach
1. **Image-based classification (Sentinel-2 data)**  
   - Uses spectral data to group pixels into meaningful clusters.
2. **Altimetry classification (Sentinel-3 data)**  
   - Uses waveform characteristics to differentiate sea ice from open water.
3. **Evaluation and Comparison**  
   - Compare clustering results with ESA’s official classification to assess accuracy.

This notebook provides a step-by-step guide to implementing these methods, from preprocessing satellite data to visualizing and interpreting clustering results. The goal is to demonstrate how unsupervised learning can be applied in real-world remote sensing applications.

---


## **K-means Clustering Example**
This section demonstrates the use of K-means clustering to classify randomly generated data into four clusters.

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

This section introduces Gaussian Mixture Models (GMM), which are a probabilistic model for representing normally distributed subpopulations within an overall dataset. Unlike K-means clustering, which assigns each data point to a single cluster, GMM provides a probability-based clustering approach, meaning each point has a probability of belonging to multiple clusters. This makes it more flexible for identifying clusters with different shapes and variances.

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

In this section, we apply K-Means clustering to classify Sentinel-2 satellite imagery, specifically distinguishing between sea ice and open water. The goal is to leverage unsupervised learning to detect differences in spectral characteristics of Sentinel-2 bands. This method helps automate image classification, which is useful for environmental monitoring and climate research.

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

The displayed image represents the K-Means clustering classification applied to Sentinel-2 bands, where the algorithm has grouped pixels into distinct clusters based on their spectral values.

#### **What this shows:**
The image highlights two different clusters (represented by different colors), distinguishing features within the Sentinel-2 satellite imagery. This clustering approach helps in identifying variations in surface features, such as differentiating between sea ice and leads in the dataset.

#### **Why this is useful:**
- It provides an unsupervised classification method to analyze remote sensing imagery.
- Helps in automating feature detection in satellite images, reducing manual interpretation.
- Supports further analysis for environmental monitoring and climate studies.


# **Gaussian Mixture Model (GMM) Clustering on Sentinel-2 Bands**

## **Introduction**
In this section, we apply the Gaussian Mixture Model clustering algorithm to Sentinel-2 imagery. Unlike K-Means, which assigns data points to the nearest cluster center, GMM uses probabilistic clustering, allowing for soft classification of pixels. This method is useful in remote sensing when spectral properties of land, water, and ice overlap.

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

- The color-coded regions correspond to different clusters identified by the Gaussian Mixture Model (GMM) based on reflectance values from the Sentinel-2 imagery.
- The dark purple region (left side) represents areas where the algorithm could not classify data due to missing or low-reflectance values.
- The yellow and green regions indicate different terrain features, with the yellow areas likely corresponding to highly reflective surfaces such as ice or snow-covered regions.
- The blue-green areas might represent water bodies or darker terrain, which have lower reflectance in the selected band.
- The gradual color transition shows how GMM captures soft boundaries between different surface types, unlike K-Means which enforces hard clustering.

This result is useful for **remote sensing applications**, helping to distinguish between features such as **ice, open water, and land** using unsupervised learning.

# Altimetry Classification

Now, let's explore the application of unsupervised methods to altimetry classification tasks, focusing specifically on distinguishing between sea ice and leads in the Sentinel-3 altimetry dataset.

Before we can apply classification methods, we must preprocess the data by extracting meaningful features. The following functions are used for this purpose.

---

## **1. Preprocessing Functions for Feature Extraction**

These functions **prepare the dataset** by computing waveform properties, ensuring proper interpolation of variables, and calculating statistical features.

### **Peakiness Calculation**
This function determines the **peakiness of waveform signals**, helping differentiate between different surface types in the Sentinel-3 dataset.

```python
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy.ma as ma
import glob
from matplotlib.patches import Polygon
import scipy.spatial as spatial
from scipy.spatial import KDTree
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster

def peakiness(waves, **kwargs):
    "Finds peakiness of waveforms."
    import numpy as np
    import matplotlib.pyplot as plt

    print("Running peakiness function...")

    size = np.shape(waves)[0]
    waves1 = np.copy(waves)

    if waves1.ndim == 1:
        print('Only one waveform in file')
        waves1 = waves1.reshape(1, np.size(waves1))

    def by_row(waves, *args):
        "Calculate peakiness for each waveform"
        maximum = np.nanmax(waves)
        if maximum > 0:
            maximum_bin = np.where(waves == maximum)[0][0]
            waves_128 = waves[maximum_bin-50:maximum_bin+78]
            noise_floor = np.nanmean(waves_128[10:20])
            where_above_nf = np.where(waves_128 > noise_floor)

            if np.shape(where_above_nf)[1] > 0:
                maximum = np.nanmax(waves_128[where_above_nf])
                mean = np.nanmean(waves_128[where_above_nf])
                peaky = maximum / mean
            else:
                peaky = np.nan
        else:
            peaky = np.nan

        if 'peaky' in args:
            return peaky

    peaky = np.apply_along_axis(by_row, 1, waves1, 'peaky')

    return peaky

def unpack_gpod(variable):

    from scipy.interpolate import interp1d

    time_1hz=SAR_data.variables['time_01'][:]
    time_20hz=SAR_data.variables['time_20_ku'][:]
    time_20hzC = SAR_data.variables['time_20_c'][:]

    out=(SAR_data.variables[variable][:]).astype(float)  # convert from integer array to float.

    #if ma.is_masked(dataset.variables[variable][:]) == True:
    #print(variable,'is masked. Removing mask and replacing masked values with nan')
    out=np.ma.filled(out, np.nan)

    if len(out)==len(time_1hz):

        print(variable,'is 1hz. Expanding to 20hz...')
        out = interp1d(time_1hz,out,fill_value="extrapolate")(time_20hz)

    if len(out)==len(time_20hzC):
        print(variable, 'is c band, expanding to 20hz ku band dimension')
        out = interp1d(time_20hzC,out,fill_value="extrapolate")(time_20hz)
    return out


#=========================================================================================================
#=========================================================================================================
#=========================================================================================================

def calculate_SSD(RIP):

    from scipy.optimize import curve_fit
    # from scipy import asarray as ar,exp
    from numpy import asarray as ar, exp

    do_plot='Off'

    def gaussian(x,a,x0,sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    SSD=np.zeros(np.shape(RIP)[0])*np.nan
    x=np.arange(np.shape(RIP)[1])

    for i in range(np.shape(RIP)[0]):

        y=np.copy(RIP[i])
        y[(np.isnan(y)==True)]=0

        if 'popt' in locals():
            del(popt,pcov)

        SSD_calc=0.5*(np.sum(y**2)*np.sum(y**2)/np.sum(y**4))
        #print('SSD calculated from equation',SSD)

        #n = len(x)
        mean_est = sum(x * y) / sum(y)
        sigma_est = np.sqrt(sum(y * (x - mean_est)**2) / sum(y))
        #print('est. mean',mean,'est. sigma',sigma_est)

        try:
            popt,pcov = curve_fit(gaussian, x, y, p0=[max(y), mean_est, sigma_est],maxfev=10000)
        except RuntimeError as e:
            print("Gaussian SSD curve-fit error: "+str(e))
            #plt.plot(y)
            #plt.show()

        except TypeError as t:
            print("Gaussian SSD curve-fit error: "+str(t))

        if do_plot=='ON':

            plt.plot(x,y)
            plt.plot(x,gaussian(x,*popt),'ro:',label='fit')
            plt.axvline(popt[1])
            plt.axvspan(popt[1]-popt[2], popt[1]+popt[2], alpha=0.15, color='Navy')
            plt.show()

            print('popt',popt)
            print('curve fit SSD',popt[2])

        if 'popt' in locals():
            SSD[i]=abs(popt[2])


    return SSD
```
```
path = '/content/drive/MyDrive/AL4EO/W4/Unsupervised Learning/'
SAR_file = 'S3A_SR_2_LAN_SI_20190307T005808_20190307T012503_20230527T225016_1614_042_131______LN3_R_NT_005.SEN3'
SAR_data = Dataset(path + SAR_file + '/enhanced_measurement.nc')

SAR_lat = unpack_gpod('lat_20_ku')
SAR_lon = unpack_gpod('lon_20_ku')
waves   = unpack_gpod('waveform_20_ku')
sig_0   = unpack_gpod('sig0_water_20_ku')
RIP     = unpack_gpod('rip_20_ku')
flag = unpack_gpod('surf_type_class_20_ku')

# Filter out bad data points using criteria (here, lat >= -99999)
find = np.where(SAR_lat >= -99999)
SAR_lat = SAR_lat[find]
SAR_lon = SAR_lon[find]
waves   = waves[find]
sig_0   = sig_0[find]
RIP     = RIP[find]

# Calculate additional features
PP = peakiness(waves)
SSD = calculate_SSD(RIP)

# Convert to numpy arrays (if not already)
sig_0_np = np.array(sig_0)
PP_np    = np.array(PP)
SSD_np   = np.array(SSD)

# Create data matrix
data = np.column_stack((sig_0_np, PP_np, SSD_np))

# Standardize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)
```

# **Altimetry Classification - Preprocessing Functions**

## **Overview**
Before applying machine learning models to classify Sentinel-3 altimetry data, it is essential to preprocess the raw waveform data. This step ensures that the input features are properly structured and optimized for classification. The following preprocessing functions help transform waveform characteristics into meaningful numerical values that can be used to distinguish between sea ice and leads.

## **Peakiness Calculation**
The peakiness function analyzes the shape of a waveform by identifying the highest peak and measuring how sharply the waveform energy is concentrated. By computing the peak-to-mean ratio, this function provides insight into waveform sharpness, which is useful for distinguishing between surfaces like open water and sea ice. Leads generally exhibit higher peakiness, while rougher ice surfaces have more spread-out energy distributions.

## **GPOD Variable Extraction**
Different variables in the Sentinel-3 dataset exist at different resolutions, requiring interpolation for consistency. The GPOD extraction function resamples low-resolution 1 Hz data to match the 20 Hz high-resolution measurements, ensuring that all features are aligned correctly. This step is critical for preventing discrepancies in time-series analysis and ensuring the dataset is complete for classification models.

## **Stack Standard Deviation (SSD) Calculation**
The SSD function analyzes the spread of waveform energy by fitting a Gaussian model to the waveform. This provides a numerical measure of how dispersed the energy is, which can help distinguish between smooth open water and rough sea ice surfaces. A higher SSD value generally indicates a more scattered waveform, which is associated with ice-covered regions, while lower values suggest smoother, more reflective surfaces such as leads.

## **Conclusion**
These preprocessing functions play a crucial role in preparing Sentinel-3 altimetry data for classification tasks. Peakiness quantifies waveform sharpness, GPOD variable extraction ensures data consistency, and SSD provides insight into waveform spread. Together, these steps refine the dataset, making it suitable for unsupervised learning models that classify sea ice and open water.


There are some NaN values in the dataset so one way to deal with this is to delete them.

```
# Remove any rows that contain NaN values
nan_count = np.isnan(data_normalized).sum()
print(f"Number of NaN values in the array: {nan_count}")

data_cleaned = data_normalized[~np.isnan(data_normalized).any(axis=1)]

mask = ~np.isnan(data_normalized).any(axis=1)
waves_cleaned = np.array(waves)[mask]
flag_cleaned = np.array(flag)[mask]

data_cleaned = data_cleaned[(flag_cleaned==1)|(flag_cleaned==2)]
waves_cleaned = waves_cleaned[(flag_cleaned==1)|(flag_cleaned==2)]
flag_cleaned = flag_cleaned[(flag_cleaned==1)|(flag_cleaned==2)]
```

Now, let's proceed with running the GMM model as usual. You can also replace it with K-Means or any other model of your choice.

```
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data_cleaned)
clusters_gmm = gmm.predict(data_cleaned)
```
We can also examine the number of data points in each class of the clustering prediction.

```
unique, counts = np.unique(clusters_gmm, return_counts=True)
class_counts = dict(zip(unique, counts))
print("Cluster counts:", class_counts)
```

We can plot the mean waveform of each class.

```
# mean and standard deviation for all echoes
mean_ice = np.mean(waves_cleaned[clusters_gmm==0],axis=0)
std_ice = np.std(waves_cleaned[clusters_gmm==0], axis=0)

plt.plot(mean_ice, label='ice')
plt.fill_between(range(len(mean_ice)), mean_ice - std_ice, mean_ice + std_ice, alpha=0.3)


mean_lead = np.mean(waves_cleaned[clusters_gmm==1],axis=0)
std_lead = np.std(waves_cleaned[clusters_gmm==1], axis=0)

plt.plot(mean_lead, label='lead')
plt.fill_between(range(len(mean_lead)), mean_lead - std_lead, mean_lead + std_lead, alpha=0.3)

plt.title('Plot of mean and standard deviation for each class')
plt.legend()
```

## Clustering Output Image

Below is the clustering result:

![Clustering Output](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/Unknown.png?raw=true)

### **Mean and Standard Deviation of Waveforms for Each Class**
The plot above visualizes the mean waveform along with the standard deviation for each class identified in the clustering process. The two classes, "ice" and "lead," are represented by distinct colors.

- The solid lines show the average waveform shape for each class.
- The shaded regions indicate the variability within each class, represented by the standard deviation.

This visualization helps in understanding the differences between waveforms associated with ice and lead, showing distinct peaks and variations that contribute to their classification.

## **Plotting All Echoes**

The following code generates a plot displaying all the echoes in the dataset.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate X values (time steps for each echo)
x = np.stack([np.arange(1, waves_cleaned.shape[1] + 1)] * waves_cleaned.shape[0])

# Plot all echoes
plt.plot(x, waves_cleaned)
plt.title("Plot of All Echoes")
plt.xlabel("Time Steps")
plt.ylabel("Echo Amplitude")
plt.show()
```

## **Plot of All Echoes**

The plot below visualizes all echoes in the dataset:

![Plot of All Echoes](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/np.stack.png?raw=true)


## **Plot of Echoes for the Lead Cluster**

The following code generates a plot displaying echoes for the **lead cluster**.

```python
import numpy as np
import matplotlib.pyplot as plt

# Extract echoes for the lead cluster
x = np.stack([np.arange(1, waves_cleaned[clusters_gmm == 1].shape[1] + 1)] * waves_cleaned[clusters_gmm == 1].shape[0])

# Plot echoes
plt.plot(x, waves_cleaned[clusters_gmm == 1])
plt.title("Plot of Echoes for the Lead Cluster")
plt.xlabel("Time Steps")
plt.ylabel("Echo Amplitude")
plt.show()
```

### **Plot of Echoes for the Lead Cluster**

![Plot of Echoes for the Lead Cluster](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/lead%20cluster.png?raw=true)

## **Plot of Echoes for the Sea Ice Cluster**

The following code generates a plot displaying echoes for the **sea ice cluster**.

```python
import numpy as np
import matplotlib.pyplot as plt

# Extract echoes for the sea ice cluster
x = np.stack([np.arange(1, waves_cleaned[clusters_gmm == 0].shape[1] + 1)] * waves_cleaned[clusters_gmm == 0].shape[0])

# Plot echoes
plt.plot(x, waves_cleaned[clusters_gmm == 0])
plt.title("Plot of Echoes for the Sea Ice Cluster")
plt.xlabel("Time Steps")
plt.ylabel("Echo Amplitude")
plt.show()
```

## **Plot of Echoes for the Sea Ice Cluster**

![Plot of Echoes for the Sea Ice Cluster](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/leadcluster.png?raw=true)

    

## **Scatter Plots of Clustered Data**

This code visualizes the clustering results using scatter plots, where different colors represent different clusters (`clusters_gmm`).

```python
import matplotlib.pyplot as plt

# Scatter plot for sig_0 vs PP
plt.scatter(data_cleaned[:, 0], data_cleaned[:, 1], c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("PP")
plt.show()

# Scatter plot for sig_0 vs SSD
plt.scatter(data_cleaned[:, 0], data_cleaned[:, 2], c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("SSD")
plt.show()

# Scatter plot for PP vs SSD
plt.scatter(data_cleaned[:, 1], data_cleaned[:, 2], c=clusters_gmm)
plt.xlabel("PP")
plt.ylabel("SSD")
plt.show()
```

## **Scatter Plots of Clustered Data**

These scatter plots visualize the clustering results using different feature pairings. Each color represents a different cluster determined by the `clusters_gmm` algorithm.

---

### **1. Scatter Plot: sig_0 vs PP**
This scatter plot shows the relationship between `sig_0` and `PP`, with clusters distinguished by color.

![Scatter Plot: sig_0 vs PP](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/scatter%201.png?raw=true)

---

### **2. Scatter Plot: sig_0 vs SSD**
This scatter plot visualizes how `sig_0` relates to `SSD`, with clear clustering patterns.

![Scatter Plot: sig_0 vs SSD](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/scatter%202.png?raw=true)

---

### **3. Scatter Plot: PP vs SSD**
This scatter plot compares `PP` and `SSD`, showing how the clusters are distributed based on these two features.

![Scatter Plot: PP vs SSD](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/scatter%203.png?raw=true)

## **Waveform Alignment Using Cross-Correlation**

This section explains how waveforms in the cluster where `clusters_gmm == 0` are aligned using cross-correlation. Cross-correlation helps to shift the waveforms so that they align to a reference point, improving clustering consistency.

---

### **Code Explanation**
1. **Import Required Function**  
   - `correlate` from `scipy.signal` is used for computing cross-correlation.

2. **Find the Reference Point**  
   - The reference point is chosen as the **peak** of the average waveform in the cluster.

3. **Align Waveforms Using Cross-Correlation**  
   - Each waveform is cross-correlated with the reference to determine the **shift** needed for alignment.
   - `np.roll()` is used to **shift** the waveforms accordingly.

4. **Plot the Aligned Waveforms**  
   - The aligned waveforms are plotted to visualize the effect of cross-correlation.

---

### **Python Code for Waveform Alignment**

```python
from scipy.signal import correlate
import numpy as np
import matplotlib.pyplot as plt

# Find the reference point (e.g., the peak)
reference_point_index = np.argmax(np.mean(waves_cleaned[clusters_gmm == 0], axis=0))

# Calculate cross-correlation with the reference point
aligned_waves = []
for wave in waves_cleaned[clusters_gmm == 0][:len(waves_cleaned[clusters_gmm == 0]) // 10]:
    correlation = correlate(wave, waves_cleaned[clusters_gmm == 0][0])
    shift = len(wave) - np.argmax(correlation)
    aligned_wave = np.roll(wave, shift)  # Shift waveform to align
    aligned_waves.append(aligned_wave)

# Plot aligned waves
for aligned_wave in aligned_waves:
    plt.plot(aligned_wave)

plt.title('Plot of 10 equally spaced functions where clusters_gmm = 0 (aligned)')
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.show()
```
### **Waveform Alignment Visualization**

This plot shows the **aligned waveforms** after applying **cross-correlation** to the `clusters_gmm == 0` group.

![Waveform Alignment Plot](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/function.png?raw=true)


## Compare with ESA Data

The ESA dataset classifies sea ice as 1 and lead as 2. To align our predicted labels with the ESA dataset, we subtract 1 from the ESA labels. This ensures:
- Sea Ice (1 in ESA) is mapped to 0 in our classification.
- Lead (2 in ESA) is mapped to 1 in our classification.

This allows us to directly compare our GMM-predicted clusters with ESA’s official labels.

---

### Python Code for Evaluation
To assess our classification, we compute:
1. Confusion Matrix – Measures how well our model assigns correct labels.
2. Classification Report – Includes precision, recall, and F1-score to evaluate performance.

```python
# Adjust ESA dataset labels for comparison
flag_cleaned_modified = flag_cleaned - 1  # Convert ESA labels to match predicted labels

from sklearn.metrics import confusion_matrix, classification_report

true_labels = flag_cleaned_modified  # True labels from the ESA dataset
predicted_gmm = clusters_gmm         # Predicted labels from GMM method

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_gmm)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Compute classification report
class_report = classification_report(true_labels, predicted_gmm)

# Print classification report
print("\nClassification Report:")
print(class_report)
```

Confusion Matrix:
[[8856   22]
 [  24 3293]]


Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      8878
         1.0       0.99      0.99      0.99      3317

    accuracy                           1.00     12195
   macro avg       1.00      1.00      1.00     12195
weighted avg       1.00      1.00      1.00     12195

## Conclusion and Results Explanation

### Summary of Findings
The Gaussian Mixture Model (GMM) classification was evaluated against the ESA official classification using a confusion matrix and classification report. The results show that the model performs exceptionally well in distinguishing between sea ice and lead.

- Out of 12,195 total cases, only **46 were misclassified**.
- The model correctly identified 8,856 instances of sea ice** and 3,293 instances of lead.
- Precision, recall, and F1-scores for both classes were close to 1.00, indicating near-perfect classification.

### Key Takeaways
1. **High Accuracy** – The model achieved an accuracy of **99.6%**, meaning it correctly classified nearly all instances.
2. **Minimal Errors** – Only **22 sea ice cases** were misclassified as lead, and **24 lead cases** were misclassified as sea ice.
3. **Reliable Performance** – The classification results closely match ESA’s official product, proving the GMM approach is effective for echo classification.

### Limitations and Considerations
While the model performed well, there are a few aspects to consider:
- Some misclassifications still exist, which could be further analyzed to understand their cause.
- Alternative clustering methods, such as supervised learning, could be explored for comparison.
- The dataset used is highly structured, and performance may vary with more diverse or noisy data.

### Final Conclusion
The results demonstrate that the GMM-based classification is highly reliable for distinguishing sea ice from leads. The model’s performance is closely aligned with ESA’s official classification, making it a valid approach for echo classification in altimetry datasets.




