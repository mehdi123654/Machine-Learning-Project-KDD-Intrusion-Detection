# KDD Intrusion Detection Model

This repository contains the implementation of several machine learning models for detecting network intrusions using the KDD Cup 1999 dataset. It includes data preprocessing, feature selection, model training, evaluation, and clustering for anomaly detection.

## Project Overview

The objective of this project is to classify network traffic as normal or one of several types of attacks. The project is divided into the following steps:
1. **Loading Data:** The KDDTrain+ and KDDTest+ datasets are loaded for training and testing the models.
2. **Data Preprocessing:** Includes encoding categorical variables, handling outliers, and normalizing data for improved model performance.
3. **Feature Selection:** Feature selection techniques like Recursive Feature Elimination (RFE) and Principal Component Analysis (PCA) are used to select the most relevant features.
4. **Modeling:** Various classification models are used, including:
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest
   - Voting Classifier (Ensemble Learning)
5. **Clustering & Anomaly Detection:** K-Means and Gaussian Mixture Models (GMM) are applied to detect anomalous network traffic.
6. **Evaluation Metrics:** Models are evaluated using confusion matrices, accuracy, precision, recall, F-measure, ROC curves, and AUC.

## Dataset

The datasets used in this project are:
- **KDDTrain+**: Training data for the model.
- **KDDTest+**: Test data used for evaluating model performance.

The dataset contains the following labels:
- `normal`: Normal network traffic.
- Various attack types categorized as DoS, Probe, R2L, and U2R.

## Data Preprocessing

The preprocessing steps include:
- **Handling Missing Values**: There are no missing values in the dataset.
- **Outliers Detection**: Boxplots are used to detect outliers. Outliers are identified but not removed.
- **Categorical Data Encoding**: One-hot encoding is used for categorical features such as `protocol_type`, `service`, and `flag`.
- **Scaling**: Features are standardized using `StandardScaler` to normalize the data for SVM, KNN, and other models.

## Feature Selection

- **RFE (Recursive Feature Elimination)** is used to select the top features for each type of attack.
- **PCA (Principal Component Analysis)** is used to reduce dimensionality and improve model performance.

## Models

### 1. **Support Vector Machine (SVM)**
   - SVM is used with linear kernels.
   - High accuracy, precision, and recall values are observed for DoS and Probe attack detection.

### 2. **K-Nearest Neighbors (KNN)**
   - KNN performs well with high accuracy and balanced precision-recall scores across all attack types.

### 3. **Logistic Regression**
   - Logistic regression models are trained for each type of attack.
   - Good overall accuracy but slightly lower performance on U2R attacks.

### 4. **Decision Tree Classifier**
   - Decision Trees show exceptional performance for DoS and Probe attacks.
   - The decision trees are visualized to show the paths taken to classify each instance.

### 5. **Random Forest & Voting Classifier (Ensemble Learning)**
   - Random forests and a voting classifier combining SVM, KNN, and Decision Trees achieve robust results with strong generalization.

## Clustering & Anomaly Detection

- **K-Means**: Applied to the PCA-transformed dataset. The elbow method is used to find the optimal number of clusters.
- **Gaussian Mixture Model (GMM)**: GMM clustering is used to fit normally distributed clusters. Voting schemes are applied to classify anomalous traffic.

## Evaluation Metrics

- **Confusion Matrix**: Provides insights into true positives, false positives, true negatives, and false negatives.
- **Accuracy, Precision, Recall, F1-Score**: Used to evaluate model performance.
- **ROC Curves & AUC**: ROC curves visualize model performance, with high AUC values for most models.

## Visualizations

The project includes various plots for exploratory data analysis, such as:
- Histograms and boxplots for feature distribution.
- Count plots to analyze the relationship between `protocol_type` and attacks.
- Pie charts showing the distribution of attacks in the training and test sets.
- PCA scatter plots and cluster visualizations.

## Results Summary

- The SVM model provided the highest precision and recall for DoS and Probe attacks.
- The Decision Tree model showed excellent performance, especially in detecting unauthorized access (R2L).
- Clustering with K-Means and GMM demonstrated good separation between normal and anomalous traffic.

## Future Work

Further improvements can include:
- Hyperparameter tuning for improved model accuracy.
- Deep learning models such as neural networks for better generalization.
- Exploration of additional clustering methods for enhanced anomaly detection.

---

Feel free to update any specific sections as needed for your project!
