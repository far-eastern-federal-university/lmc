import pandas as pd
from sklearn.linear_model import LogisticRegression
from cross_validation_plotter import plot_cross_validation
from sklearn.model_selection import cross_val_score

data = pd.read_csv("wdbc.data", sep = ",")
data = data.iloc[:,0:12]
#print(data.describe())
X = data.iloc[:,2:12]
y = data.iloc[:,1]

# Забираем рабочий код у ирисов

clf = LogisticRegression(random_state=123, solver='lbfgs',
                         multi_class='multinomial',  max_iter=300)

clf.fit(X, y)

print("Where is malignant cancer?")
n = 1
print(clf.predict(X[14:23]))
print((X[14:23]))
print(cross_val_score(clf, X, y, cv=10))

# Забираем функцию для построения графиков (не забываем подгрузить cross_validation_plotter)
print("---------------------------------------")
help(plot_cross_validation) # 
print("---------------------------------------")
print("Вызов функции")
param=plot_cross_validation(X=X, y=y, clf=clf, title="Logistic Regression")
print("---------------------------------------")
print(param)