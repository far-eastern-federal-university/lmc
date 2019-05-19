#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# импортируем набор данных (iris)
iris = datasets.load_iris()
X = iris.data[:, :2] # возьмём только первые 2 признака, чтобы проще воспринять вывод
y = iris.target

C = 1.0 # параметр регуляризации SVM
svc = svm.SVC(kernel='linear', C=1,gamma=0.1).fit(X, y) # здесь мы взяли линейный kernel

# создаём сетку для построения графика
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
 np.arange(y_min, y_max, 0.1))

plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()


# In[13]:





# In[14]:


sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
sns.regplot('X','Y', data=df, logistic=True)
plt.ylabel('Probability')
plt.xlabel('Explanatory')


# In[ ]:




