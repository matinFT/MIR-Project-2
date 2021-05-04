import typing as th
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# since you can use sklearn (or other libraries) implementations for this task,
#   you can either initialize those implementations in the provided format or use them as you wish


class SVM(BaseEstimator, ClassifierMixin):
    def __init__(self,kernel = 'rbf',C = 1,gamma = 'auto',degree = 3):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.degree = degree
        self.clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma, degree=degree))
        
    def fit(self, X, Y):
        new_X = np.array([np.array(x) for x in X])
        new_Y = np.array(Y)
        self.clf.fit(new_X, new_Y)

    def predict(self, X):
        new_X = np.array([np.array(x) for x in X])
        return self.clf.predict(new_X)
        