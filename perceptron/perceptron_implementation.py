# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    
    
    """
    " Param {float} alpha (Learning rate)
    " Param {int} epoch (Iterations)
    " param {int} random_seed (Seed)
    """
    def __init__(self, alpha = 0.01, epoch = 50, random_seed = 1):
        self.alpha = alpha
        self.epoch = epoch
        self.random_seed = random_seed
        
    """
    " Param {array} X
    " Param {arrat} y
    """
    def fit(self, X, y):
        random = np.random.RandomState(self.random_seed)
        self.w_ = random.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.errors_ = []
        
        for i in range(self.epoch):
            errors = 0
            for xi, target in zip(X,y):
                update = self.alpha * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
#            print('---------weight array---------')
#            print(self.w_)
#            print('---------error array---------')
#            print(self.errors_)
        return self
    
    """
    " Param {array} X
    """
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    """
    " Param {int} X
    """
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
def plot_decision_regions(X, y, classifier, resolution = 0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha=0.8, c = colors[idx], marker = markers[idx], label = cl, edgecolor = 'black')
    
dataFrame = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
dataFrame.tail()
# Extracting only 2 labels i.e. iris-setosa and versicolor which are the first 100 samples
y = dataFrame.iloc[0:100, 4].values
# Replacing the labels with 1 and -1 for perceptron
y = np.where(y == 'Iris-setosa', -1, 1)
# Extracting only 2 features i.e. sepal length and petal length
X = dataFrame.iloc[0:100, [0,2]].values

# Plotting the data
plt.scatter(X[:50, 0], X[:50,1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.legend(loc='upper left')
plt.show()

# fitting the data to perceptron
perceptron = Perceptron(alpha=0.1, epoch=10)
perceptron.fit(X, y)
plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

# plotting the colored regions
plot_decision_regions(X, y, classifier = perceptron)
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.legend(loc = 'upper left')
plt.show()