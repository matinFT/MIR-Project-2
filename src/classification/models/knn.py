import typing as th
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class KNN(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            k: int,
            # add required hyper-parameters (if any)
    ):
        self.k = k

#     def fit(self, x, y, **fit_params):

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
        
        nearest_neighbors_indices = [i for i in range(self.k)]
        nearest_neighbors_distances = [sum(np.power(self.X[i] - x, 2)) for i in range(self.k)]
        for i in range(self.k, len(self.X)):
            distance = sum(np.power(self.X[i] - x, 2))
            replace_index = -1
            for k in range(self.k):
                if distance < nearest_neighbors_distances[k]:
                    if replace_index == -1 or distance < nearest_neighbors_distances[replace_index]:
                        replace_index = k
                        
            if replace_index != -1:
                nearest_neighbors_indices[replace_index] = i
                nearest_neighbors_distances[replace_index] = distance
        return [self.Y[i] for i in nearest_neighbors_indices]
     
    
                        
