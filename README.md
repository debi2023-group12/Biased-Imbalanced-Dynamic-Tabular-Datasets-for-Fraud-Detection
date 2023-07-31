# Biased-Imbalanced-Dynamic-Tabular-Datasets-for-Fraud-Detection

This repository contains code for building and evaluating a fraud detection model. The model is designed to predict fraudulent activities based on a dataset containing various features related to customer behavior and transactions. The code leverages different machine learning algorithms, such as Logistic Regression, XGBoost, Random Forest, and Gradient Boosting Machine (GBM), to create and tune the fraud detection model.

## Table of Contents

1. Introduction
2. Installation
3. Usage
4. Code Overview
5. Methods

## 1. Introduction

Fraud detection is a critical task for many businesses to prevent financial losses and maintain customer trust. The goal of this project is to build a machine learning model that can effectively identify fraudulent activities based on historical data. The code provided here offers a flexible and customizable pipeline for preprocessing, training, and evaluating the fraud detection model using various machine learning algorithms.

## 2. Installation

To run the code in this repository, you need to have Python and the required dependencies installed. The primary dependencies include:

- numpy
- pandas
- aequitas
- xgboost
- sklearn
- tensorflow
- imblearn
- pycaret
- missingno
- association_metrics
- scipy
- scikeras
- cuml

You can install these dependencies using pip:

pip install numpy pandas aequitas xgboost sklearn tensorflow imblearn pycaret missingno association_metrics scipy scikeras cuml

### 3. Usage

To use the fraud detection model, follow these steps:

1. Download the dataset from this link: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022.
1. Clone the repository to your local machine.
2. Import the 'FraudDetectionModel' class from the provided Python script into your code.
3. Instantiate the 'FraudDetectionModel' class and load your dataset into it. Please, make sure you enter the dataset path correctly.
4. Choose the desired preprocessing option (e.g., 'baseline', 'option1', 'option2') to handle missing values, feature engineering, and other preprocessing tasks.
5. Train the model using one of the available algorithms (Logistic Regression, XGBoost, Random Forest, GBM, or Deep Neural Network).
6. Evaluate the model's performance on test data using metrics like accuracy, precision, recall, F1-score, and ROC curve.
7. Optionally, tune hyperparameters for better model performance.

Note: Examples of using the package are available in the notebooks (Step1.ipynb, Step2_3.ipynb, and Step2_3_4.ipynb)

## 4. Code Overview

The code is organized into different methods and classes, each serving a specific purpose in the fraud detection pipeline. Here's an overview of the main components:

#### Bank_Account_Fraud_Detection: 
The main class responsible for preprocessing, training, evaluating, and tuning the fraud detection model.
#### Preprocessing: 
Methods to handle missing values, perform feature engineering, one-hot encoding, and standard scaling.
#### Model Training: 
Methods to train the model using Logistic Regression, XGBoost, Random Forest, GBM, and Deep Neural Network.
#### Evaluation: 
Methods to evaluate the model's performance, including confusion matrix, ROC curve, and various metrics.
#### EDA: 
Method for exploritory data analysis

## 5. Methods

#### load_data: 
Loads the data file using the file name.
#### preprocess: 
Handles missing values, feature engineering, one-hot encoding, and standard scaling.
#### fit_baseline, tune_baseline_LR, tune_baseline_XGB, tune_basline_RF, tune_baseline_DNN: 
Train and tweak the hyperparameters for the baseline models.
#### train_logistic, train_XGB, train_RF, train_DNN, train_GBM: 
Train the model using different algorithms.
#### evaluate_GBM, evaluate: 
Evaluate the performance of the model using various metrics.
#### tune_baseline_LR, tune_baseline_XGB, tune_baseline_DNN, tune_baseline_RF, tune_GBM: 
Tune hyperparameters for better model performance.
#### plot_GBM, plot_confusion: 
Plot visualization for GBM model, confusion matrix, and ROC curve.
