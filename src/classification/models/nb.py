import typing as th  # Literals are available for python>=3.8
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class NaiveBayes(BaseEstimator, ClassifierMixin):
    
    def __init__(self,
            kind  #: th.Literal['gaussian', 'bernoulli', ],
            # add required hyper-parameters (if any)
    ):
        self.kind = kind

#     def fit(self, x, y, **fit_params):
        
    def fit(self, x, y):
        if self.kind == "gaussian":
            self.classes_num = max(y)
            self.class_priors = self.cal_class_priors(y)
            self.means, self.stds = self.cal_means_variances(x, y)
    
    def cal_means_variances(self, X, Y): # calculates means of gaussian p(t|c)
        means = np.zeros([len(X[0]), self.classes_num + 1])
        stds = np.zeros([len(X[0]), self.classes_num + 1])
        for i in range(len(X[0])):
            for c in range(len(means[0])):
                features = []
                for j in range(len(X)):
                    if Y[j] == c:
                        features.append(X[j][i])
                means[i][c] = np.mean(features)
                stds[i][c] = np.std(features)
        return means, stds
                    
    def cal_class_priors(self, y):
        classes_num = [0] * (self.classes_num + 1)
        for i in y:
            classes_num[i] += 1
        n = len(y)
        return [classes_num[i] / n for i in range(len(classes_num))]
    
    def predict(self, X):
        if self.kind == "gaussian":
            return self.pred_gaussian(X)
        else:
            return pred_bernouli(X)
    
    def pred_gaussian(self, X):
        Y = []
        constant = 1 / np.sqrt(2 * np.pi)
        for x in X:
            class_probs = []
            for c in range(self.classes_num + 1):
                term_probs = [constant / self.stds[i][c] * np.exp(-(x[i] - self.means[i][c])**2 / 2 / self.stds[i][c]**2) for i in range(len(x))]
                prob = np.log(self.class_priors[c])
                for p in term_probs:
                    prob += np.log(p)
                class_probs.append(prob)
            Y.append(np.argmax(class_probs))
        return Y
    
    def pred_bernouli(self, X):
        pass
                                                                                