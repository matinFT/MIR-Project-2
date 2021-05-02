import typing as th
from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
import numpy as np

class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    def __init__(
            self,
            k,
            iterations,
    ):
        self.k = k
        self.iterations = iterations
        self.centroids = None

    def fit(self, x):
        X = np.array([np.array(i) for i in x])
        centroids = X[np.random.choice(len(X), self.k, replace = False)].copy()
        for i in range(self.iterations):
            Y = self.points_new_classes(X, centroids)
            centroids = self.new_centroids(X, Y)
        self.centroids = centroids
        self.labels = Y
    
    
    def points_new_classes(self, X, centroids):
        classes = np.zeros(len(X))
        distances = np.zeros(self.k)
        for i in range(len(X)):
            x = X[i]
            for c in range(len(distances)):
                distances[c] = sum((x - centroids[c]) ** 2)
            classes[i] = int(np.argmin(distances))
        return classes
    
    def new_centroids(self, X, Y):
        class_sums = np.zeros([self.k, len(X[0])])
        class_total = np.zeros(self.k)
        for i in range(len(X)):
            
            class_sums[int(Y[i])] += X[i]
            class_total[int(Y[i])] += 1
        return np.array([class_sums[i] / class_total[i] for i in range(len(class_total))])
        
    def predict(self, X):
        
        Y = np.zeros(len(X))
        for i in range(len(X)):
            
            
            Y[i] = np.argmin([sum((X[i] - centroid) ** 2) for centroid in self.centroids])
        return Y
