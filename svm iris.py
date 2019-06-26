from cross_validation_plotter import plot_cross_validation
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score


# импортируем набор данных (iris)
iris = datasets.load_iris()
X = iris.data[:, :2] # возьмём только первые 2 признака, чтобы проще воспринять вывод
y = iris.target

C = 1.0 # параметр регуляризации SVM
svc = svm.SVC(kernel='linear', C=1, gamma=0.1)
svc.fit(X, y)


print("---------------------------------------")
print("Предсказание класса для первых двух объектов из X")
print(svc.predict(X[:2, :]))
print("---------------------------------------")
print(cross_val_score(svc, X, y, cv=10))

# Забираем функцию для построения графиков (не забываем подгрузить cross_validation_plotter)
print("---------------------------------------")
help(plot_cross_validation) # 
print("---------------------------------------")
print("Вызов функции")
param= plot_cross_validation(X=X, y=y, clf=svc, title="SVM")
print("---------------------------------------")
print(param)
print("---------------------------------------")
