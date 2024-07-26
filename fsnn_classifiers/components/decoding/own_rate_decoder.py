import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class OwnRateDecoder(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        if hasattr(self, '_validate_data'):
            # requires sklearn>=0.23
            X, y = self._validate_data(X, y, ensure_2d=True)
        else:
            X, y = check_X_y(X, y)
            self.n_features_in_ = X.shape[0]
        self.classes_ = unique_labels(y)

        self.mean_train_rates_ = [
                np.mean(X[y == current_class][:,current_class])
                for current_class in self.classes_
            ]

        self.mean_train_rates_ = np.asarray(self.mean_train_rates_)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, 'mean_train_rates_')

        probe = np.repeat(self.mean_train_rates_[np.newaxis, :], len(X), axis=0)

        votes_for_classes = np.abs(X - probe)

        y_predicted = np.argmin(votes_for_classes, axis=-1)

        return y_predicted
