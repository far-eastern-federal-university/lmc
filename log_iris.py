# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:09:28 2019

@author: Dmitry
"""

from cross_validation_plotter import plot_cross_validation
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#--------------------------------------------------------------
# Блок загрузки данных

iris = load_iris()

X, y = load_iris(return_X_y=True)

clf = LogisticRegression(random_state=123, solver='lbfgs',
                         multi_class='multinomial',  max_iter=300)

clf.fit(X, y)

print("---------------------------------------")
print("Предсказание класса для первых двух объектов из X")
print(clf.predict(X[:2, :]))
print("---------------------------------------")

print("---------------------------------------")
print("Доля ошибок классификации, при разных разбиениях X для cross-validation")
print(cross_val_score(clf, X, y, cv=10))
print("---------------------------------------")

print("---------------------------------------")
help(plot_cross_validation) # 
print("---------------------------------------")
print("Вызов функции")
plot_cross_validation(X=X, y=y, clf=clf, title="Logistic Regression")
print("---------------------------------------")