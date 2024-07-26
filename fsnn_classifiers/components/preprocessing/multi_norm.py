import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler

class MultiNorm(BaseEstimator, TransformerMixin):
    def __init__(self, type, **kwargs):
        self.type = type
        if self.type.lower() in ['l1', 'l2', 'max']:
            self.norm = Normalizer(self.type.lower(), copy=kwargs.get('copy', True))
        elif self.type.lower() == "ss":
            self.norm = StandardScaler(copy=kwargs.get('copy', True),
                                       with_mean=kwargs.get('with_mean', True),
                                       with_std=kwargs.get('with_std', True)
                                       )
        elif self.type.lower() == "mms":
            self.norm = MinMaxScaler(feature_range=kwargs.get('feature_range', (0.,1.)),
                                     copy=kwargs.get('copy', True),
                                     clip=kwargs.get('clip', False)
                                     )
        else:
            raise NotImplementedError("This norm type is not supported.")

    def fit(self, X, y=None):
        X = check_array(X)

        self.norm.fit(X, y)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = check_array(X)
        check_is_fitted(self, ['is_fitted_'])

        return self.norm.transform(X)
