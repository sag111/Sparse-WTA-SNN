import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.linear_model import LogisticRegression

class MultiDecoder(BaseEstimator, ClassifierMixin):
    def __init__(self, mod: tuple):
        self.mod = mod
        key, val = self.mod
        if key == "LogisticRegression":
            self.fun = LogisticRegression(**val)
        else:
            raise NotImplementedError("This transform type is not supported.")
        
    def fit(self, X, y):
        self.fun.fit(X, y)

    def predict(self, X):
        return self.fun.predict(X)
