#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection  import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()


# Question 1
# Convert the sklearn.dataset `cancer` to a DataFrame. 
def answer_one():
    dataframe = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    dataframe['target'] = cancer.target
    
    return dataframe


# Question 2
# What is the class distribution? (i.e. how many instances of `malignant` (encoded 0) and how many `benign` (encoded 1)?)
def answer_two():   
    cancerdf = answer_one()
    index = ['malignant', 'benign']
    
    malignants = np.where(cancerdf['target'] == 0.0)
    benings = np.where(cancerdf['target'] == 1.0)
    data = [np.size(malignants), np.size(benings)]
    
    series = pd.Series(data, index=index)
    
    return series


# Question 3
# Split the DataFrame into `X` (the data) and `y` (the labels).
def answer_three():   
    df = answer_one()
    
    X = df[df.keys()[:len(df.keys())-1]]
    y = df['target']
        
    return (X ,y)

# Question 4
# Using `train_test_split`, split `X` and `y` into training and test sets `(X_train, X_test, y_train, and y_test)`.
def answer_four():
    X, y = answer_three()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        
    return X_train, X_test, y_train, y_test


# Question 5
# Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with `X_train`, `y_train` and using one nearest neighbor (`n_neighbors = 1`).
def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
        
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_train, y_train)
    
    return knn


# Question 6
# Using your knn classifier, predict the class label using the mean value for each feature.
def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    
    knn = answer_five()

    return knn.predict(means)    


# Question 7
# Using your knn classifier, predict the class labels for the test set `X_test`.
def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    predict_lst = knn.predict(X_test)
    return predict_lst


# Question 8
# Find the score (mean accuracy) of your knn classifier using `X_test` and `y_test`.
def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    score = knn.score(X_test, y_test)
    
    return score

