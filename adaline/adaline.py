# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions

class Adaline(object):
    
    
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
        self.cost_ = []
        
        for i in range(self.epoch):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            self.w_[1:] += self.alpha * X.T.dot(errors)
            self.w_[0] += self.alpha * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    """
    " Param {array} X
    """
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    """
    " Param {array} X
    """
    def activation(self, X):
        return X
    
    
    """
    " Param {int} X
    """
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
"""    
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
"""
dataFrame = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
print(dataFrame.tail())
# Extracting only 2 labels i.e. iris-setosa and versicolor which are the first 100 samples
y = dataFrame.iloc[0:100, 4].values
# Replacing the labels with 1 and -1 for perceptron
y = np.where(y == 'Iris-setosa', -1, 1)
# Extracting only 2 features i.e. sepal length and petal length
X = dataFrame.iloc[0:100, [0,2]].values

# Standardizing the features for better performance
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

"""
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
"""
"""
Below is the code without standardization
"""
#fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,4))
#ada1 = Adaline(epoch = 10, alpha = 0.01).fit(X,y)
#print(range(1, len(ada1.cost_) + 1))
#print('ada1.cost',ada1.cost_)
#print('log cost ada1.cost',np.log10(ada1.cost_))
#ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
#ax[0].set_xlabel('Epochs')
#ax[0].set_ylabel('log(Sum-Squared-error)')
#ax[0].set_title('Adaline - Learning rate = 0.01')
#
#ada2 = Adaline(epoch = 10,alpha = 0.0001).fit(X,y)
#print('ada2.cost_',ada2.cost_)
#print('log cost ada1.cost',np.log10(ada2.cost_))
#ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
#ax[1].set_xlabel('Epochs')
#ax[1].set_ylabel('log(Sum-Squared-error)')
#ax[1].set_title('Adaline - Learning rate = 0.0001')

"""
Below is with standardization
"""
ada = Adaline(epoch=15, alpha=0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, clf=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [Standardized]')
plt.ylabel('petal length [Standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1,len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared-Error')
plt.show()