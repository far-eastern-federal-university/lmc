# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:20:00 2019

@author: Клим
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from cross_validation_plotter import plot_cross_validation
from sklearn.svm import SVC

data = pd.read_csv("titanic.csv", sep = ",")

data_notna = pd.DataFrame.dropna(data)

X = data_notna(["Pclass", "Fare", "Age", "Sex"])
X.replace("male", 0, True, None, False)
X.replace("female", 1, True, None, False)

y = data_notna["Survived"]

clf = 
