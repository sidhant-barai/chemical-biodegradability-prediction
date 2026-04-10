# chemical-biodegradability-prediction
# Chemical Biodegradability Prediction (Machine Learning)

## 📌 Project Overview
This project involves developing a predictive model to determine the biodegradability of chemicals based on their structural properties (QSAR data). By utilizing machine learning, we provide a computational alternative to traditional animal testing in pharmacology and environmental science.

---

## 🚀 The STAR Breakdown

### **S - Situation**
Determining chemical biodegradability is a critical but resource-intensive process in environmental safety. I worked with a dataset of 1,055 chemicals, each featuring 41 distinct structural descriptors, to automate this assessment.

### **T - Task**
My goal was to design, implement, and validate a machine learning pipeline in Python that could accurately classify chemicals as biodegradable or non-biodegradable.

### **A - Action**
* **Exploratory Data Analysis (EDA):** Used Seaborn and Matplotlib to visualize data distributions via heatmaps and box plots.
* **Data Cleaning:** Removed outliers using a Z-score threshold of 3 to ensure high data quality.
* **Feature Engineering:** Applied **Principal Component Analysis (PCA)** to reduce the feature space from 41 to 26 components while maintaining 99% of the variance.
* **Model Development:** Developed and compared **Random Forest Classification** and **Regression** models.
* **Validation:** Employed **5-fold cross-validation** to ensure model stability and prevent overfitting.

### **R - Result**
* Achieved a peak predictive accuracy of **81.1%** using the Random Forest Classifier.
* Demonstrated that classification models significantly outperform regression for discrete environmental safety targets.
* Produced a professional technical report following the **IEEE Transactions format**.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Scikit-Learn, Pandas, Seaborn, Matplotlib
- **Techniques:** PCA, Random Forest, Cross-Validation, Z-Score Outlier Detection

## 📂 Repository Contents
- `dm230118267.py`: Master Python script for the full ML pipeline.
- `dm230118267.pdf`: Comprehensive IEEE-formatted research report.
