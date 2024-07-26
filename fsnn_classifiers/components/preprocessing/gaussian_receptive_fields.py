import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

def get_mu(min_x, max_x, field_number, n_fields):
    mu_step = (max_x - min_x) / (n_fields - 1)
    return min_x + mu_step*field_number

def get_sigma_squared(min_x, max_x, n_fields):
    # as in [Yu et al. 2014]
    return (
        2/3 * (max_x - min_x) / (n_fields - 2)
    )**2

def get_gaussian(x, sigma_squared, mu):

    result = (
        # So that the maximum value, if x == mu, is 1
        np.e ** (- (x - mu) ** 2 / (2 * sigma_squared))
    )

    return result

class GaussianReceptiveFields(BaseEstimator, TransformerMixin):
    def __init__(self, n_fields, cutoff_gaussian = 0.09):
        self.n_fields = n_fields
        self.cutoff_gaussian = cutoff_gaussian

    def fit(self, X, y=None):
        X = check_array(X)

        self.X_column_min_ = np.min(X, axis=0)
        self.X_column_max_ = np.max(X, axis=0)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = check_array(X)
        check_is_fitted(self, ['X_column_min_', 'X_column_max_'])

        n_samples, n_features = X.shape

        if self.n_fields == None:
            return X
        else:

            # repeat X along the feature axis <n_fields> times
            X = np.repeat(X[:,np.newaxis,:], self.n_fields, axis=1)

            # compute sigma values
            sigma_squared = get_sigma_squared(self.X_column_min_, 
                                            self.X_column_max_, 
                                            self.n_fields)

            sigma_squared = np.repeat(sigma_squared[np.newaxis, :], X.shape[0], axis=0)
            sigma_squared = np.repeat(sigma_squared[:,np.newaxis,:], self.n_fields, axis=1)

            sigma_squared[sigma_squared == 0] = 1.
            
            # compute mu values
            mu = [get_mu(self.X_column_min_,
                        self.X_column_max_,
                        field_number,
                        self.n_fields,
                        ) for field_number in range(self.n_fields)]

            mu = np.repeat(np.asarray(mu)[np.newaxis, :, :], X.shape[0], axis=0)

            # compute gaussian
            result = get_gaussian(x = X, sigma_squared = sigma_squared, mu = mu)
            result = np.reshape(result, (n_samples, n_features*self.n_fields))

            result[result < self.cutoff_gaussian] = 0

            return result
