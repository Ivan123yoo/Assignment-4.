

This project utilizes unsupervised learning methods, including K-means clustering and Gaussian Mixture Models (GMM), to classify satellite imagery into distinct categories of sea ice and leads. The imagery data is sourced from Sentinel-2 and Sentinel-3 satellites, which provide valuable insights into Earth’s polar regions. The purpose of this analysis is to demonstrate the capability of unsupervised learning techniques in identifying and categorizing unlabelled data based on inherent patterns.

Technical Overview

The analysis includes:

K-means Clustering: Segmenting Sentinel-2 optical data to identify distinct regions based on spectral characteristics.
Gaussian Mixture Models (GMM): Using Sentinel-3 altimetry data for probabilistic classification, improving upon the deterministic methods of K-means.
Data Preparation: The raw satellite data is preprocessed to include only relevant spectral bands and remove noise or corrupt data points, ensuring high-quality inputs for the models.
Visualization: Post-classification results are visualized to assess the accuracy and to provide intuitive graphical representations of the classified data.


Step 1: Mount Google Drive
This allows your Google Colab environment to access files stored in your Google Drive.

Step 2: Install Required data
Ensure all required Python libraries and data are installed. These libraries are necessary for handling data, performing analyses, and visualizing results.


Step 3: Load and Preprocess the Data
Load the satellite data using rasterio and preprocess it to format the data for analysis. This typically includes reading specific bands, masking non-data regions, and normalizing or scaling the data.

Step 5: Apply Machine Learning Models
Implement the K-means clustering and Gaussian Mixture Model to classify the data.

Step 6: Visualize the Results
Plot the results using matplotlib to visualize the classification outputs, helping to assess the performance of your models visually.

Step 7: Analyze and Interpret the Results
Evaluate the effectiveness of the models in classifying the types of surfaces and discuss the results based on the visual outputs and any additional statistical analyses conducted.


Results
The output consists of cluster maps and probability distributions for sea ice and leads, demonstrating the effective use of unsupervised learning in environmental satellite data analysis.

Credits

Data Sources: European Space Agency (ESA) for providing Sentinel-2 and Sentinel-3 imagery.
Supervisory Support: Provided by 
