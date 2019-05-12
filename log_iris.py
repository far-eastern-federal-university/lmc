# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:09:28 2019

@author: Dmitry
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

#--------------------------------------------------------------
# Блок определения функций
def plot_cross_validation(X, y, clf, title=None):
    """ Функция plot_cross_validation предназначена для построения
        кривых обучения для метода clf по матрице объекты-признаки X
        и вектору ответов y.
        Возвращает объект-график, относящийся к библиотеке matplotlib.pyplot
    """
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=241)
    train_sizes=np.linspace(.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=cv, n_jobs=4, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()



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