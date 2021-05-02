import typing as th
from abc import ABCMeta
from sklearn.base import DensityMixin, BaseEstimator
import numpy as np
# since you can use sklearn (or other libraries) implementations for this task,
#   you can either initialize those implementations in the provided format or use them as you wish
from sklearn.mixture import GaussianMixture


class GMM(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(
            self,
            cluster_count,
            max_iteration,
            # add required hyper-parameters (if any)
    ):
        self.cluster_count = cluster_count
        self.max_iteration = max_iteration
        self.clf = GaussianMixture(n_components=cluster_count, random_state=0, max_iter=max_iteration)

    def fit(self, X):
        self.clf.fit(np.array([np.array(x) for x in X]))

    def predict(self, X):
        return self.clf.predict(np.array([np.array(x) for x in X]))
