import sys
path_to_remove = '/s/ls4/users/romanrybka/YD/fsnn-classifiers'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)

import argparse
import pickle
import shutil
import subprocess
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
from scipy.stats import mode
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.metrics import f1_score

import os

sys.path.append('/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/')
from fsnn_classifiers.datasets.load_data import load_data
from fsnn_classifiers.optimization.hpo import adjust

sys.path.append('/s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/')
from hpo_experiment_base import HPOExperimentBase


class HPOExperiment(HPOExperimentBase):
    def __init__(self, args, exp_dir_name, synapse_model):
        super(HPOExperiment, self).__init__(args, exp_dir_name, synapse_model)

    def _init_search_space(self):
        ss = OrderedDict([
            ('encoding_method', "frequency"), # 'frequency', 'time'
            ('decode_potentials', True), 
            ('V_th', 10000),
            ('experiment_path', self.base_dir.parent.parent),
            ('synapse_model', self.synapse_model),
            ('neuron_model', 'iaf_psc_exp'), # iaf_cond_exp_adaptive
            ('resolution', 0.1),
            ('tau_s', 0.),
            ('t_ref', 0.),
            ('sigma_w', None),
            ('tau_m', 1e7),
            ('w_init', 0.),

            ('time', 300),
            ('ref_seq_interval', hp.randint('ref_seq_interval', 2, 11)),

            ('tau_minus', hp.choice('tau_minus', np.arange(1., 301, 20.))),
            ('tau_plus', hp.choice('tau_plus', np.arange(1., 301, 20.))),
            ('mu_plus', hp.choice('mu_plus', np.arange(0.01, 1.01, 0.05))),
            ('mu_minus', hp.choice('mu_minus', np.arange(0.01, 1.01, 0.05))),
            ('alpha', hp.choice('alpha', [hp.uniform('alpha_small', 0., 10.), hp.loguniform('alpha_big', 2., 8.)])),
            ('lr', hp.uniform('lr', 1e-6, 1)),
            ('shift_minus', 0.),
            ('shift_plus', hp.uniform('shift_plus', -0.5, 0.)),

            ('n_fields', None),

            ('n_estimators', self.args.n_estimators),
            ('max_features', 1.0),
            ('max_samples', hp.choice('max_samples', [0.05, 0.1, 0.2])), # hp.uniform('max_samples', 0.1, 0.25)),
        ])
        with open(self.trial_dir / 'search_space.pkl', 'wb') as f:
            pickle.dump(ss, f)
        return ss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir_name", default="unknown")
    parser.add_argument("--job_id", default="unknown")
    parser.add_argument("--ho_bound", type=int, default=10000)
    parser.add_argument("--n_estimators", type=int, default=1)
    parser.add_argument("--dir_for_resume", default="")
    parser.add_argument("--validation_size", type=int, default=1000)
    args = parser.parse_args()

    exp = HPOExperiment(args, "shifted_nn_symm_stdp_synapse", "shifted_nn_symm_stdp_synapse")
    exp.optimize()

if __name__ == "__main__":
    main()
