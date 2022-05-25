
from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]  # setosa, versicolor, verginica

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]
classifier1 = SVC(kernel="linear", C=float(99999))
classifier1.fit(X, y)
print(classifier1.coef_)
print(classifier1.intercept_)