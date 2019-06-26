import numpy as np
import pandas as pd
from cross_validation_plotter import plot_cross_validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm

#Load the dataset
df = pd.read_csv("diabetes.csv")

#Print the first 5 rows of the dataframe.
df.head()
df.shape

#создадим пустые массивы для объектов и цели
X = df.drop('Outcome',axis=1).values
y = df['Outcome'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)

#Setup для хранения точности обучения и тестирования
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

C = 1.0 # параметр регуляризации SVM
svc = svm.SVC(kernel='linear', C=1, gamma=0.1)
svc.fit(X, y)

print("have diabetes or not?")
n = 1
print(svc.predict(X[14:23]))
print((X[14:23]))
print(cross_val_score(svc, X, y, cv=10))

# Забираем функцию для построения графиков (не забываем подгрузить cross_validation_plotter)
print("---------------------------------------")
help(plot_cross_validation) # 
print("---------------------------------------")
print("Вызов функции")
param=plot_cross_validation(X=X, y=y, clf=svc, title="SVM")
print("---------------------------------------")
print(param)