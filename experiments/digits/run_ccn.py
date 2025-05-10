import os
os.environ["PYNEST_QUIET"] = "1"

import sys

path_to_remove = '/s/ls4/users/romanrybka/YD/fsnn-classifiers'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)
    
sys.path.insert(0, '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN')

from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.optimization.hpo import adjust

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

import numpy as np

from typing import OrderedDict

from hyperopt import hp

import argparse
import pickle

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def run(test_size):

    X, y = load_digits(return_X_y=True)

    # Flatten the images.
    X = X.reshape(
        (X.shape[0], -1)
    )

    params = {
        'n_fields': 8,
        'max_features': 1.0,  
        'max_samples': 1.0,
        'bootstrap_features': False, 
        'weight_normalization': None, 
        'w_inh': None, 
        'w_init': 0.0, 
        'synapse_model': 'stdp_nn_pre_centered_synapse', 
        'V_th': -65.81794269776182, 
        't_ref': 0.0, 
        'corr_time': 0.0,
        'tau_m': 30, 
        'Wmax': 1.0,
        'intervector_pause': 60,
        'mu_plus': 0.01747439420954161, 
        'mu_minus': 0.29325365965710104, 
        'random_state': None, 
        'use_entire_train_data': False, 
        'data_sampler': None, 
        'sigma_w': 0.835584173228832, 
        'lr': 0.012778485884244321, 
        'early_stopping': False, 
        'allow_offgrid_times': True, 
        'time': 1000,
        'n_jobs': 1, 
        'warm_start': False, 
        'quiet': True, 
        'sample_norm': 1,
        'I_negative': -1000.0, 
        'V_th_train': 10000.0, 
        'resolution': 0.5,  
        'epochs': 1,
        'tau_s': 3.3, 
        'I_exc': 10000000.0, 
        'ref_seq_interval': 2, 
        'correct_encoding': True,
    }

    path = '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/digits/data_for_folds/' + \
    params["synapse_model"]

    os.makedirs(path, exist_ok=True)

    grf = GRF(n_fields=params["n_fields"])
    X = grf.fit_transform(X)

    ccn = CorrelationClasswiseNetwork(**params)

    pipe = make_pipeline(ccn) 

    lr = LogisticRegression(max_iter=100000)
    # prob_dec = ProbabilityClassifier(delta_x=1, mode=0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

    pipe.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    y_pred_ccn = pipe.predict(X_test)
    y_pred_lr = lr.predict(X_test)

    ccn_score = f1_score(y_pred_ccn, y_test, average="micro")
    lr_score = f1_score(y_pred_lr, y_test, average="micro")

    print("\ntest_size: " + str(test_size))
    print("\nf1_micro max rate dec: " + str(ccn_score))
    print("\nf1_micro log regr dec: " + str(lr_score))

    # return -1 * np.mean(scores)


def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--plasticity", type=str, default="stdp_tanh_synapse")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--ho_bound", type=int, default=600)
    parser.add_argument("--job_id", type=str, default="unknown_job_id")
    parser.add_argument("--n_fields", type=int, default=30)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        run(i)