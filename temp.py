import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from cross_validation_plotter import plot_cross_validation
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

# импортируем набор данных (titanic)
data = pd.read_csv("titanic.csv", sep = ",")

data_notna = pd.DataFrame.dropna(data)
X = data_notna[["Pclass", "Fare", "Age", "Sex"]]
X.replace("male", 0, True, None, False)
X.replace("female", 1, True, None, False)

y = data_notna["Survived"]

C = 1 # параметр регуляризации SVM
svc = svm.SVC(kernel='linear', C=1, gamma=0.1)
svc.fit(X, y)


print(svc.predict(X.loc[:3, :]))

print("---------------------------------------")
print("Первые два объекта из X")
print(X.loc[:3, :])
print("Предсказание класса для первых двух объектов из X")
print(svc.predict(X.loc[:3, :]))
print("---------------------------------------")


# Забираем функцию для построения графиков (не забываем подгрузить cross_validation_plotter)
print("---------------------------------------")
help(plot_cross_validation) # 
print("---------------------------------------")
print("Вызов функции")
plot_cross_validation(X=X, y=y, clf=svc, title="SVM")
print("---------------------------------------")

print("Тадам! Всё работает")
