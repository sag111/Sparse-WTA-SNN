
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.decomposition import PCA

class MultiTransform(BaseEstimator, TransformerMixin):
    def __init__(self, mod: tuple):
        self.mod = mod
        key, val = self.mod
        if key == "PCA":
            self.fun = PCA(**val)
        else:
            raise NotImplementedError("This transform type is not supported.")

    def fit(self, X, y=None):
        X = check_array(X)

        self.fun.fit(X, y)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = check_array(X)
        check_is_fitted(self, ['is_fitted_'])

        return self.fun.transform(X)
