import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class Multiplier(BaseEstimator, TransformerMixin):
   
   def __init__(self, input_multiplication=1):
      self.inp_mul = input_multiplication

   def fit(self, X, y=None):
      X = check_array(X)
      self.is_fitted_ = True
      return self
   
   def transform(self, X):
      X = check_array(X)
      check_is_fitted(self, 'inp_mul')
      return np.repeat(X, self.inp_mul, axis=-1)
    
