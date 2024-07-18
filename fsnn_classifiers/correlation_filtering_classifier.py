from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork
from sklearn.base import BaseEstimator, TransformerMixin
from random import choice
import numpy as np
from tqdm import trange

class CorrelationFilteringClassifier(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        
        self.ccn_1 = CorrelationClasswiseNetwork(*args, **kwargs)
        
        self.ccn_2 = CorrelationClasswiseNetwork(*args, **kwargs)
        
        self.ccn_3 = CorrelationClasswiseNetwork(*args, **kwargs)

        self.quiet = kwargs.get("quiet", False)

    def fit(self, X, y):

        n_samples = len(X)

        # fit first classifier
        self.ccn_1.fit(X, y)

        # filter samples
        
        N2_idxs = []

        y_pred_1 = self.ccn_1.predict(X)

        pbar = trange(n_samples, disable=self.quiet)
        for i in range(n_samples):
            coin_flip = choice([0, 1])

            if (coin_flip == 0 and y_pred_1[i] == y[i]) or (coin_flip == 1 and y_pred_1[i] != y[i]):
                N2_idxs.append(i)
            pbar.update(1)

        pbar.close()

        sample_ratio = len(X)/len(N2_idxs)
        self.ccn_2.max_samples = min(1.0, self.ccn_2.max_samples * sample_ratio)

        self.ccn_2.fit(X[N2_idxs], y[N2_idxs])

        y_pred_2 = self.ccn_2.predict(X)

        N3_idxs = []
        # filter samples again
        pbar = trange(n_samples, disable=self.quiet)
        for i in range(n_samples):
            if y_pred_1[i] != y_pred_2[i]:
                N3_idxs.append(i)
            pbar.update(1)

        pbar.close()

        sample_ratio = len(X)/len(N3_idxs)
        self.ccn_3.max_samples = min(1.0, self.ccn_3.max_samples * sample_ratio)

        self.ccn_3.fit(X[N3_idxs], y[N3_idxs])

    def transform(self, X, y=None):
        trf_1 = self.ccn_1.transform(X)
        trf_2 = self.ccn_2.transform(X)
        trf_3 = self.ccn_3.transform(X)
        return np.concatenate([trf_1, trf_2, trf_3], axis=-1)

    def predict(self, X, y=None):
        n_samples = len(X)
        y_pred = np.zeros(len(X), dtype=np.int32)

        corr_1 = self.ccn_1.transform(X)
        corr_2 = self.ccn_2.transform(X)
        corr_3 = self.ccn_3.transform(X)

        corr = corr_1 + corr_2 + corr_3



        pbar = trange(n_samples, disable=self.quiet)
        for i, s in enumerate(corr):
            y_pred[i] = self.ccn_3._most_frequent_class(np.argmax(s, axis=1))
            pbar.update(1)
        pbar.close()

        return y_pred




