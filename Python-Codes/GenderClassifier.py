from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
import numpy as np


# Data and labels
X = [[181, 80, 44], [170, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42],
     [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female',
      'male', 'male']

# Classifiers
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()

# Traiinig models
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_KNN.fit(X, Y)

# Testing data
pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM is: {}'.format(acc_svm))

pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for Decision Tree is: {}'.format(acc_tree))

pred_perceptron = clf_perceptron.predict(X)
acc_perceptron = accuracy_score(Y, pred_perceptron)
print('Accuracy for perceptron is: {}'.format(acc_perceptron))

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracuy for KNN is: {}'.format(acc_KNN))

# Best Classifier
index = np.argmax([acc_KNN, acc_svm, acc_tree, acc_perceptron])
print(index)
classifiers = {0: 'KNN', 1: 'SVM', 2: 'DecisionTree', 3: 'Perceptron'}
print('Best Classifier is {}'.format(classifiers[index]))
