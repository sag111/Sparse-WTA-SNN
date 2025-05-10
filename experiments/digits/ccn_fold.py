import os
os.environ["PYNEST_QUIET"] = "1"

import sys

path_to_remove = '/s/ls4/users/romanrybka/YD/fsnn-classifiers'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)
    
sys.path.insert(0, '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN')

from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork
from fsnn_classifiers.components.preprocessing.dog import DoG
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.optimization.hpo import adjust

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import load_breast_cancer, load_digits

from fsnn_classifiers.components.decoding.probability_decoder import ProbabilityClassifier

import numpy as np

from typing import OrderedDict

from hyperopt import hp

import argparse
import pickle


def load_data(script_suffix):

    train_idxs = np.load(f'{path}/train_idxs_{script_suffix}_fold.npy')
    test_idxs = np.load(f'{path}/test_idxs_{script_suffix}_fold.npy')

    return train_idxs, test_idxs


if __name__ == "__main__":

    path = '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/digits/data_for_folds/' + sys.argv[2]

    with open(f'{path}/params.pkl', 'rb') as f:   
        params = pickle.load(f)

    # ccn_params['intervector_pause'] = ccn_params['tau_m'] * 1.5

    X, y = load_digits(return_X_y=True)
    # Flatten the images.
    # X = X.reshape(
    #     (X.shape[0], -1)
    # )

    if "sigma1" in params.keys():
        dog1 = DoG(**dog1_params)
        dog2 = DoG(**dog2_params)

        X = np.concatenate((dog1.fit_transform(X), dog2.fit_transform(X)), axis=-1)
        X = np.clip(X, 0, None)

    elif "n_fields" in params.keys():
        grf = GRF(n_fields=params["n_fields"])
        X = grf.fit_transform(X)
        # X = np.clip(X, 0, None)

    # nrm = Normalizer(norm=params["norm"]) if params["norm"] != "std" else StandardScaler()

    # X = nrm.fit_transform(X)
    
    ccn = CorrelationClasswiseNetwork(**params)

    pipe = make_pipeline(ccn) 

    train_idxs, test_idxs = load_data(sys.argv[1])

    pipe.fit(X[train_idxs], y[train_idxs])

    train_rates = pipe.transform(X[train_idxs]) 

    lr = LogisticRegression(max_iter=100000)
    # prob_dec = ProbabilityClassifier(delta_x=1, mode=0)

    lr.fit(train_rates, y[train_idxs])
    # prob_dec.fit(train_rates, y[train_idxs])

    y_pred = pipe.predict(X[test_idxs])

    test_rates = pipe.transform(X[test_idxs])

    lr_score = f1_score(lr.predict(test_rates), y[test_idxs], average="micro")
    # prob_dec_score = f1_score(prob_dec.predict(test_rates), y[test_idxs], average="micro")

    np.save(f"{path}/lr_score_fold_{sys.argv[1]}.npy", [lr_score])
    # np.save(f"{path}/prob_decode_score_fold_{sys.argv[1]}.npy", [prob_dec_score])

    score = f1_score(y_pred, y[test_idxs], average="micro")

    print(str(score))