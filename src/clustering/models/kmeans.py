import typing as th
from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
import numpy as np
from numpy import linalg as LA

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
        for i in range(len(X)):
            classes[i] = int(np.argmin([LA.norm(a) for a in (centroids - X[i])]))
        return classes
    
    def new_centroids(self, X, Y):
        centroids = np.zeros([self.k, len(X[0])])
        for i in range(self.k):
            class_members = X[Y == i]
            centroids[i] = sum(class_members) / len(class_members)
        return centroids

    def predict(self, X):
        Y = np.zeros(len(X))
        for i in range(len(X)):
            Y[i] = np.argmin([LA.norm((x), 2) for x in (self.centroids - X[i])])
        return Y
