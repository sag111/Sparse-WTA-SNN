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
from fsnn_classifiers.components.preprocessing.center_and_normalize import center_and_normalize_vec
from fsnn_classifiers.datasets.load_data import load_data

from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
import argparse
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt

import pickle

import numpy as np

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_script(script_suffix, plasticity):
    result = subprocess.run(['python', '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/cancer/run_ccn_cancer_fold.py', str(script_suffix), str(plasticity)], capture_output=True, text=True)
    print(f"Sub-script {script_suffix} finished with stdout: {result.stdout} and stderr: {result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"Sub-script {script_suffix} failed with return code {result.returncode}")
    return float(result.stdout.strip())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, default="")
    parser.add_argument("--plasticity", type=str, default="stdp_nn_restr_synapse")
    parser.add_argument("--max_train", type=int, default=60000)
    parser.add_argument("--max_test", type=int, default=10000)
    parser.add_argument("--n_estimators", type=int, default=1)
    parser.add_argument("--time", type=int, default=1000)
    parser.add_argument("--max_samples", type=float, default=1.0)
    parser.add_argument("--max_features", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--decoding", type=str, default="frequency")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log_results", action="store_true")
    parser.add_argument("--full_ds", action="store_true")
    parser.add_argument("--mean_train", action="store_true")
    parser.add_argument("--concatenate_mnist", action="store_true")
    parser.add_argument("--minimnist_tries", type=int, default=1)
    parser.add_argument("--use_entire_train_data", action="store_true")
    parser.add_argument("--clip_val", type=float, default=0.05)
    parser.add_argument("--train_samples_without_landshafts", action="store_true")
    parser.add_argument("--test_samples_without_landshafts", action="store_true")
    parser.add_argument("--second_layer", action="store_true")
    parser.add_argument("--n_fields", type=int, default=0)
    parser.add_argument("--samples_without_their_mean", action="store_true")
    parser.add_argument("--center_and_normalize", action="store_true")
    parser.add_argument("--check_5_and_8", action="store_true")
    parser.add_argument("--weight_normalization", type=int, default=None)
    args = parser.parse_args()
    return args


def run(args):
    
    parent_directory = os.path.dirname(os.path.abspath(__file__))
    print(parent_directory)

    # Полный путь к директории для job_id
    job_directory = os.path.join(parent_directory, 'results', args.job_id)

    # Создание всех вложенных директорий, если они не существуют
    os.makedirs(job_directory, exist_ok=True)
    
    ''' NC '''
    # params = {
    #     'V_th': -68.97335532110806, 
    #     'norm': 'std', 
    #     'ref_seq_interval': 65, 
    #     'synapse_model': "stdp_tanh_synapse",
    #     'sigma_w': -0.4831198243248674, 
    #     'tau_m': 108, 
    #     'tau_s': 7.892277287245161, 
    #     'w_init': 0.0008496973485923266,
    #     'n_fields': 30
    # }

    ''' PPX '''
    params = {
        'V_th': -69.6, 
        'norm': 'std', 
        'ref_seq_interval': 5, 
        'synapse_model': "stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse",
        'sigma_w': 0.0, 
        'tau_m': 10.0, 
        'tau_s': 0.3, 
        'w_init': 0.0,
        'n_fields': 25,
        'intervector_pause': 50,
    }

    ''' STDP '''
    # params = {
    #     'V_th': -69.9, 
    #     'norm': 'std', 
    #     'ref_seq_interval': 7, 
    #     'synapse_model': "stdp_nn_symm_synapse",
    #     'sigma_w': 0.0, 
    #     'tau_m': 90.0, 
    #     'tau_s': 0.5, 
    #     'w_init': 0.0,
    #     'n_fields': 25,
    #     'intervector_pause': 50,
    # }

    # nrm = Normalizer('l2')
    nrm = StandardScaler()

    grf = GRF(n_fields=params['n_fields'])

    ccn = CorrelationClasswiseNetwork(
        # n_fields=params['n_fields'],
        n_estimators=1,
        max_features=1.0,
        max_samples=1.0,
        # synapse_model=,
        epochs=1,
        corr_time=0.0,
        t_ref=0.0,
        time=1000, 
        quiet=True,
        sample_norm=1,
        w_inh=None,
        # w_init=0.0,
        weight_normalization=None,
        early_stopping=False, #True,
        n_jobs=1,
        job_directory=job_directory,
        job_id=args.job_id,
        use_entire_train_data=args.use_entire_train_data,
        record_weights=True,
        **params,
    )

    path = f'/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/cancer/data_for_folds_cancer/{ccn.synapse_model}'
    os.makedirs(path, exist_ok=True)

    pipe = make_pipeline(grf, ccn)

    n_splits=5

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    preproccessing = "std" # params['norm']

    grf_params = grf.__dict__

    ccn_params = params
    # ccn_params['intervector_pause'] = params['tau_m'] * 1.5
    ccn_params['early_stopping'] = False
    ccn_params['n_fields'] = grf.n_fields

    serialized_params = pickle.dumps((preproccessing, grf_params, ccn_params))

    with open(f'{path}/params.pkl', 'wb') as f:
        f.write(serialized_params)

    X, y = load_breast_cancer(return_X_y=True)

    all_scores = []
    all_lr_scores = []
    for _ in range(3):

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
                np.save(f'{path}/train_idxs_{i+1}_fold.npy', train_index)
                np.save(f'{path}/test_idxs_{i+1}_fold.npy', test_index)

        scores = []

        # Запуск суб-скриптов параллельно
        with ThreadPoolExecutor(max_workers=n_splits) as executor:
            futures = [executor.submit(run_script, i, ccn.synapse_model) for i in range(1, n_splits + 1)]
            
            for future in as_completed(futures):
                score = future.result()
                scores.append(score)

        all_scores.append(scores)

        lr_scores = []
        for i in range(n_splits):
            lr_scores.append(np.load(f"{path}/lr_score_fold_{i+1}.npy")[0])
        all_lr_scores.append(lr_scores)

    print("\n\n\n")
    print("max rate scores : " + str(all_scores))
    print("\n")
    print("lr scores : " + str(all_lr_scores))
   
if __name__ == "__main__":
    args = parse_args()
    run(args) 