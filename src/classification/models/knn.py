import typing as th
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from numpy import linalg as LA


class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k = 2):
        self.k = k

    def fit(self, X, Y):
        self.X = np.array([np.array(x) for x in X])
        self.Y = np.array(Y)
        self.classes_num = max(Y) + 1

    def predict(self, X):
        pred = []
        X_arr = np.array([np.array(x) for x in X])
        for x in X_arr:
            nearest_neighbors_classes = self.nearest_neighbors_classes(x)
            pred.append(np.argmax(np.bincount(nearest_neighbors_classes)))
        return pred
            
    def nearest_neighbors_classes(self, x):
        distances = np.array([LA.norm(a, 2) for a in (self.X - x)])
        return self.Y[distances.argsort()[:self.k]]
