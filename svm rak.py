import pandas as pd
from cross_validation_plotter import plot_cross_validation
from sklearn.model_selection import cross_val_score
from sklearn import svm

data = pd.read_csv("wdbc.data", sep = ",")
data = data.iloc[:,0:12]
#print(data.describe())
X = data.iloc[:,2:12]
y = data.iloc[:,1]

# Забираем рабочий код у ирисов

C = 1.0 # параметр регуляризации SVM
svc = svm.SVC(kernel='linear', C=1, gamma=0.1)
svc.fit(X, y)

print("Where is malignant cancer?")
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