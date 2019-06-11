from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from cross_validation_plotter import plot_cross_validation

digits = datasets.load_digits()

X, y = datasets.load_digits(return_X_y=True)

C = 1.0 # параметр регуляризации SVM
svc = svm.SVC(kernel='linear', C=1, gamma=0.1)
svc.fit(X, y)


print("Блок вывода информации")
print("---------------------------------------")
print("Размерность набора данных")
print(digits.images.shape)
print("---------------------------------------")

print()

print("---------------------------------------")
print("Предсказание класса для первых 20 объектов из X")
print(svc.predict(X[:20, :]))
print("---------------------------------------")

print()
print(cross_val_score(svc, X, y, cv=10))

# Забираем функцию для построения графиков (не забываем подгрузить cross_validation_plotter)
print("---------------------------------------")
help(plot_cross_validation) # 
print("---------------------------------------")
print("Вызов функции")
plot_cross_validation(X=X, y=y, clf=svc, title="SVM")
print("---------------------------------------")

print()

print("---------------------------------------")
print("Элемент из digits")
print(digits.images[0])
print("---------------------------------------")

print()

print("---------------------------------------")
print("Элемент из X")
print(X[:1, :])
print("---------------------------------------")

print()

print("---------------------------------------")
print("Визуализация digits.images[0]")
print("---------------------------------------")
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
