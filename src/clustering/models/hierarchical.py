import typing as th
from sklearn.base import ClusterMixin, BaseEstimator
# since you can use sklearn (or other libraries) implementations for this task,
#   you can either initialize those implementations in the provided format or use them as you wish
from sklearn.cluster import AgglomerativeClustering
import numpy as np

class Hierarchical(ClusterMixin, BaseEstimator):
    def __init__(
            self,
            cluster_count,
    ):
        self.cluster_count = cluster_count
        self.clf = HC_model = AgglomerativeClustering(n_clusters = 2)

    def fit_predict(self, X):
        self.clf.fit(np.array([np.array(x) for x in X]))
        return self.clf.labels_
