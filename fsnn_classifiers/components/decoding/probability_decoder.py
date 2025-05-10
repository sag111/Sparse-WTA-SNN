import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from scipy.stats import mode


class ProbabilityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, delta_x=10, mode=0, n_estimators=1):
        self.delta_x = delta_x
        self.P = None
        self.local_min = None
        self.local_max = None
        self.num_bins = None
        self.classes_ = None
        self.models = {}
        self.mode = mode
        self.n_estimators = 1


    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.classes_ = np.unique(y)
        num_classes_unique = len(self.classes_)
        
        self.local_min = np.min(X, axis=0)
        self.local_max = np.max(X, axis=0)

        # self.n_estimators = num_features / num_classes_unique
        
        self.num_bins = (np.ceil((self.local_max - self.local_min) / self.delta_x)).astype(int) + 1
        
        self.P = np.zeros((num_features, num_classes_unique, np.max(self.num_bins)))
        self.hists = np.empty((num_features, num_classes_unique), dtype=object)

        if self.mode != 4:
            for j, cls in enumerate(self.classes_):
                class_indices = np.where(y == cls)[0]
                for c in range(num_features):
                    hist, bin_edges = np.histogram(X[class_indices, c], bins=np.linspace(self.local_min[c], self.local_max[c], self.num_bins[c] + 1))
                    self.hists[c, j] = (hist, bin_edges)
                    self.P[c, j, :self.num_bins[c]] = hist / np.sum(hist)
        
        else:
            for j, cls in enumerate(self.classes_):
                class_indices = np.where(y == cls)[0]
                for c in range(num_features):
                    hist, bin_edges = np.histogram(X[class_indices, c], bins=np.linspace(self.local_min[c], self.local_max[c], self.num_bins[c] + 1))
                    self.P[c, j, :self.num_bins[c]] = hist / np.sum(hist)
                    
                    X_train = np.arange(self.num_bins[c]).reshape(-1, 1)
                    y_train = self.P[c, j, :self.num_bins[c]]
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    self.models[(c, j)] = model
                    
        return self

    
    def predict_proba(self, X):
        num_samples, num_features = X.shape
        num_classes_unique = self.P.shape[1]
        
        log_p = np.zeros((num_samples, num_classes_unique))
        for i in range(num_samples):
            for c in range(num_features): 
                for j in range(num_classes_unique): 
                    bin_index = ((X[i, c] - self.local_min[c]) / self.delta_x).astype(int)
                    
                    if self.mode == 0:
                        bin_index = np.clip(bin_index, 0, self.num_bins[c] - 1)
                        log_p[i, j] += np.log(self.P[c, j, bin_index] + 1e-9)
                    
                    elif self.mode == 1:
                        if bin_index < 0:
                            log_p[i, j] += np.log(self.P[c, j, 0] + 1e-9)
                        elif bin_index >= self.num_bins[c]:
                            log_p[i, j] += np.log(self.P[c, j, -1] + 1e-9)
                        else:
                            log_p[i, j] += np.log(self.P[c, j, bin_index] + 1e-9)
                    
                    elif self.mode == 2:
                        bin_index = np.clip(bin_index, 0, self.num_bins[c] - 1)
                        if bin_index < self.num_bins[c] - 1:
                            bin_frac = (X[i, c] - self.local_min[c]) / self.delta_x - bin_index
                            interpolated_proba = (1 - bin_frac) * self.P[c, j, bin_index] + bin_frac * self.P[c, j, bin_index + 1]
                        else:
                            interpolated_proba = self.P[c, j, bin_index]
                        log_p[i, j] += np.log(interpolated_proba + 1e-9)
                    
                    elif self.mode == 3:
                        if bin_index < 0:
                            log_p[i, j] += np.log(self.P[c, j, 0] + 1e-9)
                        elif bin_index >= self.num_bins[c] - 1:
                            log_p[i, j] += np.log(self.P[c, j, -1] + 1e-9)
                        else:
                            bin_frac = (X[i, c] - self.local_min[c]) / self.delta_x - bin_index
                            interpolated_proba = (1 - bin_frac) * self.P[c, j, bin_index] + bin_frac * self.P[c, j, bin_index + 1]
                            log_p[i, j] += np.log(interpolated_proba + 1e-9)
                    
                    elif self.mode == 4:
                        if bin_index < 0:
                            model = self.models[(c, j)]
                            predicted_proba = model.predict([[bin_index]])[0]
                            log_p[i, j] += np.log(max(predicted_proba, 0) + 1e-9)
                        elif bin_index >= self.num_bins[c] - 1:
                            model = self.models[(c, j)]
                            predicted_proba = model.predict([[bin_index]])[0]
                            log_p[i, j] += np.log(max(predicted_proba, 0) + 1e-9)
                        else:
                            bin_frac = (X[i, c] - self.local_min[c]) / self.delta_x - bin_index
                            interpolated_proba = (1 - bin_frac) * self.P[c, j, bin_index] + bin_frac * self.P[c, j, bin_index + 1]
                            log_p[i, j] += np.log(max(interpolated_proba, 0) + 1e-9)
        
        p = np.exp(log_p)

        return p


    def predict(self, X):
        num_samples, num_features = X.shape
        ensemble_predictions = np.zeros((num_samples, int(self.n_estimators)))
        num_classes_unique = len(self.classes_)

        for e in range(int(self.n_estimators)):
            feature_subset = [num_classes_unique*e + i for i in range(num_features)]
            # print(feature_subset)
            Theta = self.predict_proba(X[:, feature_subset])
            ensemble_predictions[:, e] = self.classes_[np.argmax(Theta, axis=1)]

        final_predictions = mode(ensemble_predictions, axis=1, keepdims=False)[0].flatten()
        return final_predictions

    # def predict(self, X):
    #     Theta = self.predict_proba(X)
    #     return self.classes_[np.argmax(Theta, axis=1)]