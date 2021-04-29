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
        
    def fit(self, X, Y):
        self.classes_num = max(Y)
        self.class_priors = self.cal_class_priors(Y)
        if self.kind == "gaussian":
            self.means, self.stds = self.cal_means_variances(X, Y)
        else:
            self.thresholds, self.conditional_probs = self.cal_thresholds_conditional_probs(X, Y)
            
    def cal_thresholds_conditional_probs(self, X, Y):
        conditional_probs = np.zeros([len(X[0]), self.classes_num + 1])
        total_class = np.zeros([len(X[0]), self.classes_num + 1])
        positive_class = np.zeros([len(X[0]), self.classes_num + 1])
        thresholds = sum(np.array([np.array(x) for x in X])) / len(X[0])
        for i in range(len(X)):
            for j in range(len(X[0])):
                total_class[j, Y[i]] += 1
                if (X[i][j] > thresholds[j]):
                    positive_class[j][Y[i]] += 1
        return thresholds, positive_class / total_class
                
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
            return self.pred_bernouli(X)
    
    def pred_gaussian(self, X):
        pred = []
        constant = 1 / np.sqrt(2 * np.pi)
        for x in X:
            class_probs = []
            for c in range(self.classes_num + 1):
                term_probs = [constant / self.stds[i][c] * np.exp(-(x[i] - self.means[i][c])**2 / 2 / self.stds[i][c]**2) for i in range(len(x))]
                prob = np.log(self.class_priors[c])
                for p in term_probs:
                    prob += np.log(p)
                class_probs.append(prob)
            pred.append(np.argmax(class_probs))
        return pred
    
    def pred_bernouli(self, X):
        pred = []
        for x in X:
            x_bernoulli = [0] * len(x)
            for i in range(len(x)):
                if x[i] >= self.thresholds[i]:
                    x_bernoulli[i] = 1
                    
            class_probs = []
            for c in range(self.classes_num + 1):
                term_probs = [self.conditional_probs[i][c] if x_bernoulli[i] == 1 else 1 - self.conditional_probs[i][c] for i in range(len(x_bernoulli)) ]
                prob = np.log(self.class_priors[c])
                for p in term_probs:
                    prob += np.log(p)
                class_probs.append(prob)
            pred.append(np.argmax(class_probs))
        return pred
        
        
        
        