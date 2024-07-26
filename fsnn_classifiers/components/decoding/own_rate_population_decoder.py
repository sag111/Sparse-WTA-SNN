import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from collections import Counter

class OwnRatePopulationDecoder(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        if hasattr(self, '_validate_data'):
            # requires sklearn>=0.23
            X, y = self._validate_data(X, y, ensure_2d=True)
        else:
            X, y = check_X_y(X, y)
            self.n_features_in_ = X.shape[0]
        self.classes_ = unique_labels(y)

        self.n_estimators = int(X.shape[-1]/len(self.classes_))

        out_idx = np.arange(0, self.n_estimators * len(self.classes_), 1).astype(np.int32).reshape(self.n_estimators, len(self.classes_))

        self.mean_train_rates_ = []

        for i in range(self.n_estimators):

            cls_ = np.ravel(out_idx[i, :])

            estimator_mean_train_rates_ = [
                    np.mean(X[y == current_class][:,cls_[current_class]])
                    for current_class in self.classes_
                ]
            
            self.mean_train_rates_.append(estimator_mean_train_rates_)

        self.mean_train_rates_ = np.asarray(self.mean_train_rates_) # shape = (n_estimators, n_classes)

        self.is_fitted_ = True
        return self
    
    def _most_frequent_class(self, votes):
        counter = Counter(votes)
        most_common = counter.most_common(1)
        if most_common:
            return most_common[0][0] 
        else:
            return int(np.round(np.median(votes), 0))

    def predict(self, X):
        X = check_array(X) # shape = (n_samples, n_estimators * n_classes)
        check_is_fitted(self, 'mean_train_rates_')

        X = X.reshape((X.shape[0], self.n_estimators, len(self.classes_))) # shape = (n_samples, n_estimators, n_classes)

        probe = np.repeat(self.mean_train_rates_[np.newaxis, :, :], len(X), axis=0) # shape = (n_samples, n_estimators, n_classes)
        eps = 1e-6

        votes_for_classes = np.abs((X - probe)/(probe+eps))

        y_ = np.argmin(votes_for_classes, axis=-1) # shape = (n_samples, n_estimators)

        # voting across estimators
        y_predicted = np.array([self._most_frequent_class(y_[i]) for i in range(y_.shape[0])])

        return y_predicted
