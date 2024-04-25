import os
os.environ["PYNEST_QUIET"] = "1"

import sys
sys.path.insert(0, '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN')

from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork
from fsnn_classifiers.components.decoding.own_rate_population_decoder import OwnRatePopulationDecoder
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.components.preprocessing.max_pooling import Pooling
from fsnn_classifiers.optimization.hpo import adjust
from fsnn_classifiers.datasets.load_data import load_data

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

import numpy as np

from typing import OrderedDict

from hyperopt import hp

import argparse
import pickle

def experiment(args):

    trial_dir = f"{os.getcwd()}/Sparse-WTA-SNN/experiments/hpo/trials/CCN_Yura_edition_w_inh=None_MNIST_1,5K_train_1K_test/"
    os.makedirs(trial_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data("mnist", max_train=1500) 
    
    search_space = OrderedDict([
        ('plasticity', hp.choice('plasticity', ['stdp_nn_pre_centered_synapse', 'stdp_nn_restr_synapse', 'stdp_nn_symm_synapse'])),
        ('intervector_pause', hp.choice('intervector_pause', [50, 100, 150])),
        ('tau_s', hp.choice('tau_s', [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1., 1.1, 1.2, 1.5])),
        #('corr_time', hp.choice('corr_time', [5., 10., 20., 30., 50.])),
        #('I_inh', hp.choice('I_inh', [-1000, -100, -10, -1])),
        #('t_ref', hp.choice('t_ref', [1., 2., 3., 4., 5.])),
        ('norm', hp.choice('norm', ['l2', 'max'])),
        #('n_estimators', hp.choice('n_estimators', [50, 100])),
        #('max_samples', hp.choice('max_samples', [0.5, 0.7, 0.9])),
        ('V_th', hp.uniform('V_th', -70, -45)), 
        ('mu_plus', hp.uniform('mu_plus', 0.1, 1.5)),
        ('mu_minus', hp.uniform('mu_minus', 0.1, 1.5)),
        ('ref_seq_interval', hp.randint('ref_seq_interval', 3, 20)),
        ('sigma_w', hp.uniform('sigma_w', 0.1, 0.8)),
        ('tau_m', hp.randint('tau_m', 10, 90)),
        ('time', hp.randint('time', 800, 2000)),
    ])

    def run(params):

        if params["norm"] == 'std':
            nrm = StandardScaler()
        else:
            nrm = Normalizer(norm=params['norm'])
        
        ccn = CorrelationClasswiseNetwork(
            n_fields=None,
            n_estimators=1,
            max_features=1.0,
            max_samples=1.0,
            # synapse_model=args.plasticity,
            epochs=1,
            corr_time=0.0,
            t_ref=0.0,
            # time=args.time, 
            quiet=True,
            sample_norm=1,
            w_inh=None,
            w_init=0.0,
            weight_normalization=None,
            early_stopping=True,
            n_jobs=56,
            **params,
        )

        pipe = make_pipeline(nrm, ccn) 
        pipe.fit(X_train, y_train)

        f1 = f1_score(pipe.predict(X_test[:1000]), y_test[:1000], average='micro')
        print(params)
        print(f1)

        return -1 * f1
    
    adjust(run, search_space, f"{trial_dir}/trials.pkl", h_evals=1, max_evals=args.ho_bound, resume=args.resume)

def parse_args():
    parser = argparse.ArgumentParser() 
    # parser.add_argument("--plasticity", type=str, default="stdp_nn_pre_centered_synapse")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--ho_bound", type=int, default=200)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    experiment(args)