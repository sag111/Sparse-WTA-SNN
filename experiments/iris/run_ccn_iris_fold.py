import os
os.environ["PYNEST_QUIET"] = "1"

import sys

path_to_remove = '/s/ls4/users/romanrybka/YD/fsnn-classifiers'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)
    
sys.path.insert(0, '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN')

from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork
from fsnn_classifiers.components.decoding.own_rate_population_decoder import OwnRatePopulationDecoder
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.components.preprocessing.max_pooling import Pooling
from fsnn_classifiers.optimization.hpo import adjust
from fsnn_classifiers.datasets.load_data import load_data
from fsnn_classifiers.components.preprocessing.center_and_normalize import center_and_normalize_vec

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import load_breast_cancer, load_digits, load_iris

import numpy as np

from typing import OrderedDict

from hyperopt import hp

import argparse
import pickle


def load_data(path, script_suffix):

    train_idxs = np.load(f'{path}/train_idxs_{script_suffix}_fold.npy')
    test_idxs = np.load(f'{path}/test_idxs_{script_suffix}_fold.npy')

    return train_idxs, test_idxs


if __name__ == "__main__":

    path = '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/iris/data_for_folds/' + sys.argv[2]

    with open(f'{path}/params.pkl', 'rb') as f:   
        preproccessing, grf_params, ccn_params = pickle.load(f)

    # ccn_params['intervector_pause'] = ccn_params['tau_m'] * 1.5

    X, y = load_iris(return_X_y=True)

    grf = GRF(**grf_params)

    nrm = Normalizer(norm=preproccessing) if preproccessing != "std" else StandardScaler()

    X = nrm.fit_transform(X)
    
    # ccn = CorrelationClasswiseNetwork(**ccn_params)

    ccn = CorrelationClasswiseNetwork(
        # n_fields=30,
        n_estimators=1,
        max_features=1.0,
        max_samples=1.0,
        # synapse_model="stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse",
        epochs=1,
        corr_time=0.0,
        t_ref=0.0,
        time=1000, 
        quiet=True,
        sample_norm=1,
        w_inh=None,
        # w_init=0.0,
        weight_normalization=None,
        # early_stopping=False, #True,
        n_jobs=1,
        # job_directory=job_directory,
        # job_id=args.job_id,
        # use_entire_train_data=args.use_entire_train_data,
        # record_weights=True,
        **ccn_params,
    )

    ccn = CorrelationClasswiseNetwork(
        # n_fields=30,
        n_estimators=1,
        max_features=1.0,
        max_samples=1.0,
        # synapse_model="stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse", # "stdp_tanh_synapse",
        epochs=1,
        corr_time=0.0,
        t_ref=0.0,
        time=1000, 
        quiet=True,
        sample_norm=1,
        w_inh=None,
        # w_init=0.0,
        weight_normalization=None,
        # early_stopping=False,
        n_jobs=1,
        # job_id=args.job_id,
        **ccn_params,
        # **defined_params,
    )

    pipe = make_pipeline(grf, ccn) 

    train_idxs, test_idxs = load_data(path, sys.argv[1])

    pipe.fit(X[train_idxs], y[train_idxs])

    y_pred = pipe.predict(X[test_idxs])

    score = f1_score(y_pred, y[test_idxs], average="micro")

    lr = LogisticRegression(max_iter=10000)
    # prob_dec = ProbabilityClassifier(delta_x=1, mode=0)

    train_rates = pipe.transform(X[train_idxs])

    lr.fit(train_rates, y[train_idxs])
    # prob_dec.fit(train_rates, y[train_idxs])

    test_rates = pipe.transform(X[test_idxs])

    lr_score = f1_score(lr.predict(test_rates), y[test_idxs], average="micro")
    # prob_dec_score = f1_score(prob_dec.predict(test_rates), y[test_idxs], average="micro"

    np.save(f"{path}/lr_score_fold_{sys.argv[1]}.npy", [lr_score])
    # np.save(f"{path}/prob_decode_score_fold_{sys.argv[1]}.npy", [prob_dec_score]

    print(str(score))