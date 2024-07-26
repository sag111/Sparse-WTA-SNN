import os
os.environ["PYNEST_QUIET"] = "1"

from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork
from fsnn_classifiers.components.decoding.own_rate_population_decoder import OwnRatePopulationDecoder
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.datasets import load_data
from fsnn_classifiers.optimization.hpo import adjust

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import Normalizer

import numpy as np

from typing import OrderedDict

from hyperopt import hp

import argparse

def experiment(args):

    trial_dir = f"{os.getcwd()}/experiments/hpo/trials/CCN_{args.plasticity}_miniMNIST/"
    os.makedirs(trial_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(dataset=args.dataset, 
                                                 seed=1337,
                                                 )
        

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    
    search_space = OrderedDict([
        ('V_th', hp.choice('V_th', [-69.6, -69.7, -69.8, -69.9, -69.95])),
        ('tau_s', hp.choice('tau_s', [0.1, 0.3, 0.5, 0.7, 0.9])),
        ('tau_m', hp.choice('tau_m', [10.0, 30.0, 50.0, 70.0, 90.0])),
        ('ref_seq_interval', hp.choice('ref_seq_interval', [3,5,7,9])),
        ('intervector_pause', hp.choice('intervector_pause', [50,100,150])),
        ('n_fields', hp.choice('n_fields', [None, 10, 15, 20, 25, 30])),
    ])

    def run(params):

        nrm1 = Normalizer(norm='l2')
        nrm2 = Normalizer(norm='l2')

  
        grf = GRF(n_fields=params["n_fields"])

        ccn1 = CorrelationClasswiseNetwork(
            n_fields=None,
            n_estimators=11,
            max_features=0.7,
            max_samples=0.1,
            t_ref = 0.0,
            time=1000,
            w_init=0.0,
            V_th=-69.95, 
            intervector_pause=50, 
            mu_plus=0.5, 
            mu_minus=0.25,
            ref_seq_interval=9, 
            sigma_w=-0.5,
            tau_m=40.0, 
            tau_s=0.4,
        )
        
        ccn2 = CorrelationClasswiseNetwork(
            n_estimators=1,
            max_features=1.0,
            max_samples=1.0,
            synapse_model=args.plasticity,
            epochs=1,
            time=1000,
            mu_plus=0.5,
            mu_minus=0.25,
            quiet=True,
            early_stopping=True,
            **params,
        )

        pipe = make_pipeline( 
                             nrm1,  
                             ccn1,
                             #nrm2,
                             grf, 
                             ccn2)
        result = cross_validate(pipe, X, y, cv=StratifiedKFold(n_splits=5), scoring='f1_micro')

        f1 = np.mean(result['test_score'])
        print(params)
        print(f1)
        return -1 * f1
    
    adjust(run, search_space, f"{trial_dir}/trials.pkl", h_evals=1, max_evals=args.ho_bound, resume=args.resume)

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--plasticity", type=str, default="stdp_nn_pre_centered_synapse")
    parser.add_argument("--dataset", type=str, default="mnist1000")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--ho_bound", type=int, default=100)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    experiment(args)