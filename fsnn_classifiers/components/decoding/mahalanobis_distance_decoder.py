import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.base import BaseEstimator, ClassifierMixin


class MahalanobisDecoder(BaseEstimator, ClassifierMixin):
    def __init__(self, regularization=1e-6):
        """
        Mahalanobis Decoder Classifier.

        Parameters:
        - regularization: float, regularization strength for covariance matrices.
        """
        self.regularization = regularization
        self.centroids_ = None
        self.covariances_ = None
        self.num_classes_ = None
    
    def fit(self, X, y):
        """
        Train the decoder by computing centroids and covariance matrices for each class.

        Parameters:
        - X: array-like of shape (n_samples, n_features), neuron outputs.
        - y: array-like of shape (n_samples,), class labels.

        Returns:
        - self: object
        """
        self.classes_ = np.unique(y)
        self.centroids_ = []
        self.covariances_ = []
        
        for class_label in self.classes_:
            class_mask = (y == class_label)
            class_data = X[class_mask]
            
            # Compute centroid
            centroid = np.mean(class_data, axis=0)
            self.centroids_.append(centroid)
            
            # Compute covariance matrix with regularization
            cov = np.cov(class_data.T) + np.eye(self.classes_.size) * self.regularization
            self.covariances_.append(cov)
        
        self.centroids_ = np.array(self.centroids_)
        self.covariances_ = np.array(self.covariances_)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for the input data.

        Parameters:
        - X: array-like of shape (n_samples, n_features), neuron outputs.

        Returns:
        - y_pred: array of shape (n_samples,), predicted class labels.
        """
        if self.centroids_ is None or self.covariances_ is None:
            raise ValueError("The model has not been trained yet. Call 'fit' before 'predict'.")
        
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, len(self.centroids_)))
        
        for i, sample in enumerate(X):
            for j, (centroid, cov) in enumerate(zip(self.centroids_, self.covariances_)):
                try:
                    inv_cov = np.linalg.inv(cov)
                    dist = mahalanobis(sample, centroid, inv_cov)
                    distances[i, j] = dist
                except:
                    distances[i, j] = np.inf
        
        return np.argmin(distances, axis=1)
