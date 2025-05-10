import os
os.environ["PYNEST_QUIET"] = "1"

import sys

path_to_remove = '/s/ls4/users/romanrybka/YD/fsnn-classifiers'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)
    
sys.path.insert(0, '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN')

from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork
from fsnn_classifiers.components.decoding.own_rate_population_decoder import OwnRatePopulationDecoder
from fsnn_classifiers.components.decoding.mahalanobis_distance_decoder import MahalanobisDecoder
from fsnn_classifiers.optimization.hpo import adjust
from fsnn_classifiers.datasets.load_data import load_data
from fsnn_classifiers.components.preprocessing.center_and_normalize import center_and_normalize_vec
from fsnn_classifiers.components.preprocessing.dog import DoG
from fsnn_classifiers.components.preprocessing.logistic_functions import LogisticFunctions
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.components.preprocessing.pooling import Pooling
from fsnn_classifiers.components.preprocessing.clustering import clusterize

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score

import numpy as np

from typing import OrderedDict

from hyperopt import hp

import argparse
import pickle

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import mode
import time
import shutil
import ast


def experiment(args):

    print('use_poisson is', args.use_poisson)
    print('reduce_size is', args.reduce_size)
    print('convolution_window is', args.convolution_window)
    print('decode_potentials is', args.decode_potentials)
    print('run_on_test_mnist is', args.run_on_test_mnist)
    print('encoding_method is', args.encoding_method)
    print('decoding_method is', args.decoding_method)
    print('anti_stdp is', args.anti_stdp)
    print('clustering is', args.clustering)
    print('reversed_image is ', args.reversed_image)
    print('n_estimators is', args.n_estimators)
    print('n_fields is', args.n_fields)

    # Определяем директорию для выходных данных запуска
    dir_for_data = os.path.join("/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/mnist/data_from_runs", args.job_id)
    os.makedirs(dir_for_data, exist_ok=True)

    # params = {'V_th': 10000, 'alpha': 6.821084296687785, 'clustering': True, 'convolution_window': None, 'decode_potentials': True, 'decoding_method': 'mahalanobis', 'encoding_method': 'frequency', 'lr': 1e-05, 'max_features': 1.0, 'max_samples': 0.01, 'mu_minus': 0.2893912777736674, 'mu_plus': 0.16398891994532933, 'n_estimators': 25, 'n_fields': None, 'neuron_model': 'iaf_psc_exp', 'norm': 'max', 'regularization': 1e-06, 'resolution': 0.1, 'run_on_test_mnist': False, 'sigma_w': None, 'synapse_model': 'stdp_nn_pre_centered_synapse', 't_ref': 0.0, 'tau_m': 10000000.0, 'tau_s': 0.0, 'time': 300, 'train_on_all_classes': False, 'use_poisson': False, 'w_init': 0.0}

    params = {'save_weights_after_each_train_sample': True, 'n_estimators': 1, 'max_features': 1.0, 'max_samples': 0.01, 'V_th': 10000, 'decode_potentials': True, 'alpha': 53.1071633483359, 'mu_minus': 1.162313203066896, 'mu_plus': 4.915979257980662, 'n_fields': None, 'plasticity': 'stdp_nn_pre_centered_synapse', 'ref_seq_interval': 5, 'resolution': 0.1, 'sigma_w': None, 'tau_minus': 98, 'tau_plus': 39, 'tau_s': 0.1, 'time': 500, 'train_on_all_classes': False, 'use_poisson': False, 'tau_m': 1e5, 'intervector_pause': 20}

    def run(params):

        n_classes = 10

        with open(os.path.join(dir_for_data, "params.pkl"), 'wb') as f:
            pickle.dump(params, f)

        X_train, _, y_train_untouched, y_test = load_data("mnist")

        if params.get('clustering', 0):
            # Записываем кластеры для каждого из классов
            for cl in range(n_classes):
                labels, _  = clusterize(
                    X_train[y_train_untouched == cl], 
                    method='kmeans', 
                    params={'n_clusters': params["n_estimators"], 'init': 'k-means++', 'n_init': 20, 'max_iter': 1000}
                )

                np.save(os.path.join(dir_for_data, f"clusters_of_class_{cl}.npy"), labels)

        # ждем пока python скрипты с нейронами запишут свои outputs
        missing_test_outputs = set([f'neuron_{neuron}_from_ensemble_{ensemble}_test_outputs.npy' for neuron in range(n_classes) for ensemble in range(1, params["n_estimators"] + 1)])
        missing_train_files = set([f'neuron_{neuron}_from_ensemble_{ensemble}_train_outputs.npy' for neuron in range(n_classes) for ensemble in range(1, params["n_estimators"] + 1)])
        missing_files = missing_test_outputs | missing_train_files

        while missing_files:
            files_in_directory = set(os.listdir(dir_for_data))

            found_files = missing_files & files_in_directory
            if found_files:
                missing_files -= found_files

            time.sleep(10)


        mahalanobis_votes, max_votes, min_votes = [], [], []
        
        for ensemble in range(1, params["n_estimators"] + 1):

            all_neurons_test_outputs = [np.load(os.path.join(dir_for_data, f'neuron_{neuron}_from_ensemble_{ensemble}_test_outputs.npy')) for neuron in range(10)]
            all_neurons_test_outputs = np.hstack(tuple(all_neurons_test_outputs))

            all_neurons_train_outputs = [np.load(os.path.join(dir_for_data, f'neuron_{neuron}_from_ensemble_{ensemble}_train_outputs.npy')) for neuron in range(10)]
            all_neurons_train_outputs = np.hstack(tuple(all_neurons_train_outputs))

            maha_decoder = MahalanobisDecoder(regularization=params["regularization"])
            maha_decoder.fit(all_neurons_train_outputs, y_train_untouched[:10000])

            y_pred_maha = maha_decoder.predict(all_neurons_test_outputs)

            y_pred_max = np.argmax(all_neurons_test_outputs, axis=-1)

            y_pred_min = np.argmin(all_neurons_validation_outputs, axis=-1)

            mahalanobis_votes = y_pred_maha if np.shape(mahalanobis_votes) < (1,) else np.vstack((mahalanobis_votes, y_pred_maha))
            max_votes = y_pred_max if np.shape(max_votes) < (1,) else np.vstack((max_votes, y_pred_max))
            min_votes = y_pred_min if np.shape(min_votes) < (1,) else np.vstack((min_votes, y_pred_min))

        mahalanobis_votes = mahalanobis_votes.T
        max_votes = max_votes.T
        min_votes = min_votes.T

        most_frequent_votes_mahalanobis = mode(mahalanobis_votes, axis=1).mode if params["n_estimators"] > 1 else mahalanobis_votes
        most_frequent_votes_max = mode(max_votes, axis=1).mode if params["n_estimators"] > 1 else max_votes
        most_frequent_votes_min = mode(min_votes, axis=1).mode if params["n_estimators"] > 1 else min_votes

        _, _, y_train, y_test = load_data("mnist")

        scores_mahalanobis = f1_score(most_frequent_votes_mahalanobis, y_test, average=None)
        f1_micro_mahalanobis = f1_score(most_frequent_votes_mahalanobis, y_test, average="micro")

        scores_max = f1_score(most_frequent_votes_max, y_test, average=None)
        f1_micro_max = f1_score(most_frequent_votes_max, y_test, average="micro")

        scores_min = f1_score(most_frequent_votes_min, y_test, average=None)
        f1_micro_min = f1_score(most_frequent_votes_min, y_test, average="micro")

        print(
            f"params:\n{params}\n\n"
            f"Mahalanobis Decoder\nf1_micro: {round(f1_micro_mahalanobis, 2)}\n"
            f"scores:\n{np.round(scores_mahalanobis, 2)}\n\n"
            f"Max Decoder\nf1_micro: {round(f1_micro_max, 2)}\n"
            f"scores:\n{np.round(scores_max, 2)}\n\n"
            f"Min Decoder\nf1_micro: {round(f1_micro_min, 2)}\n"
            f"scores:\n{np.round(scores_min, 2)}"
        )

        return 0
    
    run(params)

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--plasticity", type=str, default="stdp_nn_restr_synapse")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--ho_bound", type=int, default=10000)
    parser.add_argument("--job_id", type=str, default="unknown_job_id")
    parser.add_argument("--train_samples_without_landshafts", action="store_true")
    parser.add_argument("--test_samples_without_landshafts", action="store_true")
    parser.add_argument("--n_fields", action="store_true")
    parser.add_argument("--train_on_all_classes", action="store_true")
    parser.add_argument("--use_poisson", action="store_true")
    parser.add_argument("--dir_for_resume", type=str, default="")
    parser.add_argument("--reduce_size", action="store_true")
    parser.add_argument("--convolution_window", type=str, default=None)
    parser.add_argument("--decode_potentials", action="store_true")
    parser.add_argument("--run_on_test_mnist", action="store_true")
    parser.add_argument("--decoding_method", type=str, default="max")
    parser.add_argument("--encoding_method", type=str, default="frequency")
    parser.add_argument("--anti_stdp", action="store_true")
    parser.add_argument("--clustering", action="store_true")
    parser.add_argument("--reversed_image", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=1)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    experiment(args)