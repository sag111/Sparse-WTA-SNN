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
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import load_breast_cancer, load_digits

import numpy as np

from typing import OrderedDict

from hyperopt import hp

import argparse
import pickle

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_script(script_suffix, plasticity):
    result = subprocess.run(['python', '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/digits/ccn_fold.py', 
    str(script_suffix), str(plasticity)], capture_output=True, text=True)
    print(f"Sub-script {script_suffix} finished with stdout: {result.stdout} and stderr: {result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"Sub-script {script_suffix} failed with return code {result.returncode}")
    return float(result.stdout.strip())


def run(n_fields):

    X, y = load_digits(return_X_y=True)

    # Flatten the images.
    X = X.reshape(
        (X.shape[0], -1)
    )

    params = {
        'n_fields': n_fields, # 8,
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

    with open(f'{path}/params.pkl', 'wb') as f:
        pickle.dump(params, f)

    n_splits=5

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        # random_state=42
    )

    all_scores = []
    all_lr_scores = []

    for k in range(1):
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
                np.save(f'{path}/train_idxs_{i+1}_fold.npy', train_index)
                np.save(f'{path}/test_idxs_{i+1}_fold.npy', test_index)

        scores = []
        
        # Запуск суб-скриптов параллельно
        with ThreadPoolExecutor(max_workers=n_splits) as executor:
            futures = [executor.submit(run_script, i, params["synapse_model"]) for i in range(1, n_splits + 1)]
            
            for future in as_completed(futures):
                score = future.result()
                scores.append(score)

        # print(f"{k}-th trial (max rate dec): ", round(np.mean(scores), 2))

        lr_scores = [np.load(f"{path}/lr_score_fold_{i}.npy")[0] for i in range(1,6)]

        # print(f"{k}-th trial (log regr dec): ", round(np.mean(lr_scores), 2))

        all_lr_scores.append(np.mean(lr_scores))
        all_scores.append(np.mean(scores))

    # print("params:\n" + str(params))
    print("\nn_fields: ", params['n_fields'])
    print("\nmean f1_micro max rate dec: " + str(np.mean(all_scores)))
    # print("\nall f1_micro max rate dec: " + str(all_scores))
    print("\nmean f1_micro log regr dec: " + str(np.mean(all_lr_scores)))
    # print("\nall f1_micro log regr dec: " + str(all_lr_scores))

    
    # lr_scores = []
    # prob_dec_scores = []
    # for i in range(n_splits):
    #     lr_scores.append(np.load(f"{path}/lr_score_fold_{i}.npy")[0])
    #     # prob_dec_scores.append(np.load(f"{path}/prob_decode_score_fold_{i}.npy")[0])

    # print("\n\n** Log Regr on spikes **\n")
    # print(round(np.mean(lr_scores)), 2)
    # print("\n\n** Prob Decoder on spikes **\n\n")
    # print(round(np.mean(prob_dec_scores)), 2)

    return -1 * np.mean(scores)



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
    for i in [3, 6, 8, 15, 25, 35, 50]:
        run(i)