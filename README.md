
# Telco Customer Churn Project
=======

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Lisans](https://img.shields.io/badge/Lisans-MIT-green.svg)](LICENSE)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](end-to-end-telco-churn.ipynb)

## 1. Project Overview
This project aims to predict the likelihood of customer churn for a telecommunications company using machine learning. The goal is to establish early warning systems to take proactive measures before losing customers. The dataset is imbalanced, with most customers continuing their subscriptions. Therefore, models are evaluated with a focus on **recall** and **F1-score** metrics.


## 2. Data Preprocessing

- Categorical features were **encoded using Label Encoding and One-Hot Encoding**.  
- Binary features were converted to `0` and `1`.  
- Due to the imbalanced dataset, some models used `class_weight='balanced'`.  
- Data was split into training and testing sets: **72% training**, **28% testing**.

## 3. Models

Four different machine learning models were tested in this project:

1. **Logistic Regression**  
   - Both balanced and unbalanced versions were tested.  
   - Hyperparameter tuning was performed using GridSearchCV.

2. **Decision Tree Classifier**  
   - Tested in its standard form (no balanced class weighting applied).  
   - No hyperparameter tuning was performed.

3. **Random Forest Classifier**  
   - Tested in its standard form (no balanced class weighting applied).  
   - No hyperparameter tuning was performed.

4. **Support Vector Classifier (SVC)**  
   - Tested in its standard form (no balanced class weighting applied).  
   - No hyperparameter tuning was performed.

## 4. Model Performance on Imbalanced Dataset

The dataset is imbalanced with a majority of class 0 samples (1440) and a minority of class 1 samples (533).  
For imbalanced datasets, **accuracy alone can be misleading**, so we focus on **class 1 performance** (minority class) using precision, recall, and F1-score.  

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.81 | 0.70 | 0.57 | 0.63 |
| Decision Tree | 0.73 | 0.48 | 0.46 | 0.47 |
| Random Forest | 0.79 | 0.66 | 0.45 | 0.54 |
| SVC | 0.80 | 0.70 | 0.45 | 0.55 |

**Note:** Metrics are calculated for **class 1 (minority class)** to better evaluate the model's ability to correctly predict the less frequent outcomes.

## 5. Balanced Logistic Regression Performance

The Logistic Regression model was trained with `class_weight='balanced'` to handle the imbalanced dataset.  
The dataset has 1440 samples for class 0 and 533 samples for class 1.  

| Class | Precision | Recall | F1-score | Support |
|-------|----------|--------|----------|---------|
| 0     | 0.91     | 0.77   | 0.83     | 1440    |
| 1     | 0.56     | 0.79   | 0.66     | 533     |
| **Weighted Avg** | 0.82 | 0.78 | 0.79 | 1973 |

**Notes:**  
- Metrics are calculated per class and weighted average.  
- Applying `class_weight='balanced'` improved recall for the minority class (class 1) compared to the unbalanced model, though precision decreased slightly.  
- Accuracy of the model: **0.78**


## 6. Usage

Follow these steps to run the project:

1. **Clone the repository:**

```bash
git clone <repo-url>
cd telco-customer-churn-project
pip install -r requirements.txt
jupyter notebook end-to-end-telco-churn.ipynb
