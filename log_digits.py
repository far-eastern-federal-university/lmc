import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from cross_validation_plotter import plot_cross_validation
from sklearn.model_selection import cross_val_score

digits = datasets.load_digits()

X, y = datasets.load_digits(return_X_y=True)

clf = LogisticRegression(random_state=123, solver='lbfgs',
                         multi_class='multinomial',  max_iter=300)

clf.fit(X, y)


print("Блок вывода информации")
print("---------------------------------------")
print("Размерность набора данных")
print(digits.images.shape)
print("---------------------------------------")

print()

print("---------------------------------------")
print("Предсказание класса для первых 20 объектов из X")
print(clf.predict(X[:20, :]))
print("---------------------------------------")

print()
print(cross_val_score(clf, X, y, cv=10))

print("---------------------------------------")
help(plot_cross_validation) 
print("---------------------------------------")
print("Вызов функции plot_cross_validation")
param=plot_cross_validation(X=X, y=y, clf=clf, title="Logistic Regression")
print("---------------------------------------")
print(param)
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
