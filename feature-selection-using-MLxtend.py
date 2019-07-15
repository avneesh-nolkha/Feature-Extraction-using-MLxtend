#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:17:53 2019

@author: avneeshnolkha
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import train_test_split

#Loading Dataset
data = pd.read_csv('stock_data.csv',error_bad_lines=False,encoding='ASCII')

x=data.isnull().sum().tolist

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    data.values[:,:-1],
    data.values[:,-1:],
    test_size=0.2,
    random_state=42)

y_train = y_train.ravel()
y_test = y_test.ravel()
"""Ravel() is another method for reshape(-1)"""
#Creating Classifier model
classifier = LogisticRegression()

#Build step forward feature selection
sfs1 = sfs(classifier,
           k_features=5,
           forward=True,
           floating=False,
           verbose=2,
           scoring='recall',
           cv=10)

sfs2 = sfs(classifier,
           k_features=20,
           forward=True,
           floating=False,
           verbose=2,
           scoring='recall',
           cv=5)
# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train).score()
sfs2 = sfs2.fit(X_train, y_train)


#Important Features
imp_features = list(sfs1.k_feature_idx_)
print(imp_features)

imp_features = list(sfs2.k_feature_idx_)
print(imp_features)


