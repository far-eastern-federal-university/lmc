# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:07:24 2019

@author: Dmitry
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from cross_validation_plotter import plot_cross_validation
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# Забираем код у предыдущих групп
data = pd.read_csv("titanic.csv", sep = ",")

data_notna = pd.DataFrame.dropna(data)

X = data_notna[["Pclass", "Fare", "Age", "Sex"]]
X.replace("male", 0, True, None, False)
X.replace("female", 1, True, None, False)

y = data_notna["Survived"]

# Забираем рабочий код у ирисов

clf = LogisticRegression(random_state=123, solver='lbfgs',
                         multi_class='multinomial',  max_iter=300)

clf.fit(X, y)



print("---------------------------------------")
print("Первые два объекта из X")
print(X.loc[:3, :])
print("Предсказание класса для первых двух объектов из X")
print(clf.predict(X.loc[:3, :]))
print("---------------------------------------")


# Забираем функцию для построения графиков (не забываем подгрузить cross_validation_plotter)
print("---------------------------------------")
help(plot_cross_validation) # 
print("---------------------------------------")
print("Вызов функции")
plot_cross_validation(X=X, y=y, clf=clf, title="Logistic Regression")
print("---------------------------------------")

print("Тадам! Всё работает")
