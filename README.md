
# Credit Card Fraud Detection using XGBoost

## Overview

This project implements a credit card fraud detection system using XGBoost and Artificial Neural Networks (ANNs). The aim is to accurately identify fraudulent transactions from a dataset of credit card transactions. This project serves as a practical demonstration of machine learning techniques in the domain of financial fraud detection.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)

## Installation

To run this project, you will need to have Python 3.x installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- xgboost
- tensorflow (for ANNs)
- matplotlib
- seaborn

You can install the required packages using pip:

pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn


## Dataset
The dataset used in this project is the Credit Card Fraud Detection dataset. It contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced, with only 0.17% of transactions being fraudulent.
Dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Methodology

Data Preprocessing:
Load the dataset and perform exploratory data analysis (EDA).
Handle missing values and normalize the data.
Split the dataset into training and test sets.

Modeling:
Implement and train two models:
XGBoost Classifier
Artificial Neural Networks (ANN)
Use evaluation metrics like AUC-ROC, precision, recall, and F1 score to assess model performance.

Comparison:
Compare the performance of both models to determine which one is more effective at detecting fraud.

## Results
The models were evaluated based on their performance metrics, including accuracy, precision, recall, and AUC score. The results are visualized using confusion matrices and ROC curves for a clear comparison.

## Conclusion
This project highlights the effectiveness of machine learning algorithms in detecting fraudulent transactions. The XGBoost model demonstrated a strong performance, showcasing the potential of machine learning in combating financial fraud.


