# Sea Ice and Lead Classification Using Unsupervised Learning

## Overview
This project applies **unsupervised machine learning techniques** to classify **sea ice and lead** using **Sentinel-2 and Sentinel-3 satellite data**. The goal is to **automate the classification process** by leveraging **clustering algorithms** such as **K-Means** and **Gaussian Mixture Model (GMM)**.

By processing **altimetry and optical satellite data**, we explore how machine learning can be used to classify oceanic features without labeled data. The results are validated against **ESA’s official classification**.

---

## Objectives
- **Preprocess satellite data** from Sentinel-2 and Sentinel-3 missions.
- **Use K-Means and GMM clustering** to classify sea ice and leads.
- **Visualize waveform characteristics** to compare ice and lead echoes.
- **Evaluate classification accuracy** using a confusion matrix and a classification report.

---

## Approach
###  Data Preprocessing
- Sentinel-2 imagery is **processed and stacked** for clustering.
- Sentinel-3 waveform data is **filtered and cleaned** for classification.

###  Unsupervised Learning Methods
- **K-Means Clustering**: Groups data points based on spectral and waveform features.
- **Gaussian Mixture Model (GMM)**: Uses a probabilistic approach for soft clustering.

###  Visualization & Evaluation
- **Clustered waveforms** are plotted to analyze sea ice and lead characteristics.
- **Confusion matrix and classification report** compare model results with ESA’s official classification.

---

## Key Features
- **Waveform Analysis**: Extracts peakiness, SSD, and other statistical features.
- **Satellite Image Classification**: Uses unsupervised learning on Sentinel-2 optical data.
- **Altimetry-Based Classification**: Applies clustering to Sentinel-3 waveforms.
- **Performance Evaluation**: Assesses classification accuracy with ESA’s dataset.

---

## Results
- **High accuracy (99.6%)** when compared with ESA’s classification.
- **Distinct clustering of sea ice and leads**, confirming the effectiveness of the models.
- **Waveform characteristics** clearly differentiate between ice and lead clusters.

 **Sample Confusion Matrix Output:**
[[8856 22] [ 24 3293]]


 **Classification Report Summary:**
Precision: 1.00 | Recall: 1.00 | F1-score: 1.00 (Sea Ice) Precision: 0.99 | Recall: 0.99 | F1-score: 0.99 (Lead) Overall Accuracy: 99.6%


---

## How to Run the Code
### **Colab / Local Setup**
1. **Mount Google Drive** (if running in Google Colab)
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
2. **Install Dependencies**
    ```sh
    pip install rasterio
    ```

3. **Run the Notebook**
   - Execute the notebook step by step.
   - The classification results will be generated automatically.

---

## Repository Structure


---

## Sample Visualizations
### **K-Means Clustering Output**
![K-Means Clustering](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/kmeans_clustering_output.png?raw=true)

### **GMM Clustering Output**
![GMM Clustering](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/Gaussian%20mixture%20model.png?raw=true)

### **Waveform Analysis**
![Waveform](https://github.com/Ivan123yoo/Assignment-4./blob/main/images/np.stack.png?raw=true)

---

## Conclusion
This project successfully demonstrates how **unsupervised learning can be applied to classify sea ice and leads** using Sentinel-2 and Sentinel-3 data. The **high classification accuracy (99.6%)** shows that clustering techniques like **K-Means and GMM** can effectively group similar oceanic features.

These methods offer a scalable, automated approach for **remote sensing and environmental monitoring**, reducing the need for manual classification.

---

## References
- European Space Agency (ESA) Sentinel Data: [ESA Sentinel Hub](https://www.sentinel-hub.com/)
- [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html) for K-Means & GMM
- Unsupervised Learning Methods: {cite}`bishop2006pattern`


