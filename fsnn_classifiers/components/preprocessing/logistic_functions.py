import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


def log_func_weights(N=None, P=None, i=None, r=None, A=0.3, B=5.9):
    for j in range(P):
        if j==0:
            yield A*np.sin(i*np.pi/(N*B)) 
            prev = A*np.sin(i*np.pi/(N*B))
        else:
            yield 1-r*prev**2 
            prev=1-r*prev**2

class LogisticFunctions(BaseEstimator, TransformerMixin):
    def __init__(self, P, r, A=0.3, B=5.9):
        self.P = P
        self.r = r
        self.A = A
        self.B = B

    def fit(self, X, y=None):
        X = check_array(X)
        self.is_fitted_ = True

        return self

    def transform(self, X):
        X = check_array(X)
        N = np.shape(X)[-1]
        weights = np.array([list(log_func_weights(N=N, P=self.P, i=i, r=self.r)) for i in range(N)])
        # weights -= weights.min()
        
        return np.dot(X, weights)

