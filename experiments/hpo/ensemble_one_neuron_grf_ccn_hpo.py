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
    print('decode_potentials is', args.decode_potentials)
    print('run_on_test_mnist is', args.run_on_test_mnist)
    print('encoding_method is', args.encoding_method)
    print('decoding_method is', args.decoding_method)
    print('anti_stdp is', args.anti_stdp)
    print('clustering is', args.clustering)
    print('reversed_image is ', args.reversed_image)
    print('n_estimators is', args.n_estimators)
    print('n_fields is', args.n_fields)
    print('simulation time is', args.time)
    print('every_input_spikes is', args.every_input_spikes)

    # Определяем базовую директорию для Trials экспериментов
    base_dir = f"/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/hpo/trials"

    # Создаем директорию для текущего задания
    trial_dir = os.path.join(base_dir, args.job_id)
    os.makedirs(trial_dir, exist_ok=True)

    # Если указана директория для возобновления, копируем файл trials.pkl
    if args.dir_for_resume:
        search_space_path = f"/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/hpo/data_for_one_neuron/{args.dir_for_resume}/search_space.pkl"
        src_path = os.path.join(base_dir, args.dir_for_resume, "trials.pkl")
        dst_path = os.path.join(trial_dir, "trials.pkl")
        
        if os.path.exists(src_path):  # Проверяем, существует ли исходный файл
            shutil.copy(src_path, dst_path)
        else:
            print(f"Файл {src_path} не найден, копирования не произошло, начинается чистый подбор.")
    
    
    if args.dir_for_resume and os.path.exists(search_space_path):
        print(f'Using pickled search_space from the previous optimization with job_id = {args.dir_for_resume}')
        with open(search_space_path, "rb") as f:
            search_space = pickle.load(f)

    else:
        print('using search_space from the script')
        search_space = OrderedDict([
            ('train_on_all_classes', args.train_on_all_classes),
            ('run_on_test_mnist', args.run_on_test_mnist),
            ('decode_potentials', args.decode_potentials),
            ('encoding_method', args.encoding_method), # 'frequency', 'time'
            ('decoding_method', args.decoding_method), # 'max', 'linregr', 'logregr', 'mahalanobis'
            ('clustering', args.clustering),
            ('reversed_image', args.reversed_image),
            ('use_poisson', args.use_poisson),
            ('every_input_spikes', args.every_input_spikes),

            ('synapse_model', 'stdp_nn_pre_centered_synapse'),
            ('neuron_model', 'iaf_psc_exp'), # iaf_cond_exp_adaptive

            ('V_th', 10000),

            ('time', int(args.time)),
            
            ('resolution', 0.1),
            ('tau_s', 0.),
            ('t_ref', 0.),
            ('sigma_w', None),
            ('tau_m', 1e7),
            ('w_init', 0.),

            ('tau_minus', hp.choice('tau_minus', np.arange(1., 2 * int(args.time), 10.))),
            ('tau_plus', hp.choice('tau_plus', np.arange(1., 2 * int(args.time), 10.))),
            ('mu_plus', hp.choice('mu_plus', np.arange(0.01, 1.01, 0.05))),
            ('mu_minus', hp.choice('mu_minus', np.arange(0.01, 1.01, 0.05))),
            ('alpha', hp.randint('alpha', 0, 25)),
            ('lr', hp.uniform('lr', 1e-6, 1)),

            ('n_fields', None),
            ('regularization', 1e-6),

            ('n_estimators', args.n_estimators),
            ('max_features', 1.0),
            ('max_samples', 0.1),
        ])

        if args.n_fields:
            search_space['n_fields'] = hp.randint('n_fields', 3, 11)

        if search_space['n_estimators'] == 1:
            search_space['max_features'] = 1

        if args.anti_stdp:
            search_space['anti_mu_plus'] = search_space['mu_plus']
            search_space['anti_mu_minus'] = search_space['mu_minus']
            search_space['anti_alpha'] = search_space['alpha']
            search_space['anti_lr'] = 0 # 0, 1e-5, hp.uniform('anti_lr', 0., 1.)

        if search_space["encoding_method"] == "time":
            search_space["time"] = hp.choice("time", [20, 50, 100])
            # search_space["epochs"] = hp.choice("epochs", [1, 3, 5])
            search_space["epochs"] = 1

    path = os.path.join('/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/hpo/data_for_one_neuron', args.job_id)
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "search_space.pkl"), "wb") as f:
        pickle.dump(search_space, f)


    def run(params):

        # Некоторые параметры
        validation_size = 1000
        n_classes = 10

        params['intervector_pause'] = params['tau_m'] * 1.1 if not args.decode_potentials else 0.
        if args.anti_stdp:
            search_space['anti_lr'] *= search_space['lr']

        with open(os.path.join(path, 'params.pkl'), 'wb') as f:
            pickle.dump(params, f)

        # Запуск скрипта, который вызывать SLURM задачи с нейронами
        try:
            result = subprocess.run(
                ["/bin/bash",
                 "/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/hpo/run_subscripts.sh",
                 str(args.job_id), str(args.n_estimators)],
                check=True  # Включаем проверку на ненулевой код завершения
            )
        except Exception as e:
            print(f"Ошибка при выполнении скрипта: {e}")
            exit(1)  # Завершаем Python скрипт с ошибкой

        X_train, _, y_train_untouched, y_test = load_data("mnist")

        if params['clustering']:
            # Записываем кластеры для каждого из классов
            for cl in range(n_classes):
                labels, _  = clusterize(
                    X_train[y_train_untouched == cl], 
                    method='kmeans', 
                    params={'n_clusters': params["n_estimators"], 'init': 'k-means++', 'n_init': 20, 'max_iter': 1000}
                )

                np.save(os.path.join(path, f'clusters_of_class_{cl}.npy'), labels)

        # ждем пока python скрипты с нейронами запишут свои outputs
        # missing_train_files = set([f'neuron_{neuron}_from_ensemble_{ensemble}_train_outputs.npy' for neuron in range(10) for ensemble in range(1, params["n_estimators"] + 1)])
        missing_validation_files = set([f'neuron_{neuron}_from_ensemble_{ensemble}_validation_outputs.npy' for neuron in range(10) for ensemble in range(1, params["n_estimators"] + 1)])
        missing_files = missing_validation_files # | missing_train_files

        while missing_files:
            # Получаем список всех файлов в директории
            files_in_directory = set(os.listdir(path))

            # Проверяем, какие файлы из нужных уже появились
            found_files = missing_files & files_in_directory
            if found_files:
                missing_files -= found_files

            # Через 10 сек проверяем еще раз
            time.sleep(10)

        # удаляем файл с параметрами чтобы запущенные задачи с сетками ждали новых параметров
        os.remove(os.path.join(path, 'params.pkl'))

        votes = None
        
        for ensemble in range(1, params["n_estimators"] + 1):

            if params["run_on_test_mnist"]:
                all_neurons_test_outputs = [np.load(os.path.join(path, f'neuron_{neuron}_from_ensemble_{ensemble}_test_outputs.npy')) for neuron in range(10)]
                all_neurons_test_outputs = np.hstack(tuple(all_neurons_test_outputs))

                if params["decoding_method"] == 'mahalanobis':
                    all_neurons_train_outputs = [np.load(os.path.join(path, f'neuron_{neuron}_from_ensemble_{ensemble}_train_outputs.npy')) for neuron in range(10)]
                    all_neurons_train_outputs = np.hstack(tuple(all_neurons_train_outputs))
                    maha_decoder = MahalanobisDecoder(regularization=params["regularization"])
                    maha_decoder.fit(all_neurons_train_outputs, y_train_untouched[:10000])
                    y_pred_mahalanobis = maha_decoder.predict(all_neurons_test_outputs)

                elif params["decoding_method"] == 'max':
                    y_pred_max = np.argmax(all_neurons_test_outputs, axis=-1)

                elif params["decoding_method"] == 'min':
                    y_pred_min = np.argmin(all_neurons_validation_outputs, axis=-1)

            else:
                all_neurons_validation_outputs = [np.load(os.path.join(path, f'neuron_{neuron}_from_ensemble_{ensemble}_validation_outputs.npy')) for neuron in range(10)]
                all_neurons_validation_outputs = np.hstack(tuple(all_neurons_validation_outputs))

                if params["decoding_method"] == 'mahalanobis':
                    maha_decoder = MahalanobisDecoder(regularization=params["regularization"])
                    maha_decoder.fit(all_neurons_validation_outputs, y_train_untouched[-int(validation_size):])
                    y_pred_mahalanobis = maha_decoder.predict(all_neurons_validation_outputs)
                
                elif params["decoding_method"] == 'max':
                    y_pred_max = np.argmax(all_neurons_validation_outputs, axis=-1)

                elif params["decoding_method"] == 'min':
                    y_pred_min = np.argmin(all_neurons_validation_outputs, axis=-1)


            if params["decoding_method"] == 'mahalanobis':
                y_pred = y_pred_mahalanobis 
                
            elif params["decoding_method"] == 'max':
                y_pred = y_pred_max
            
            elif params["decoding_method"] == 'min':
                y_pred = y_pred_min

            votes = y_pred if np.shape(votes) < (1,) else np.vstack((votes, y_pred))

        votes = votes.T

        most_frequent_votes = mode(votes, axis=1).mode if params["n_estimators"] > 1 else votes

        _, _, y_train, y_test = load_data("mnist")

        if not params["run_on_test_mnist"]:
            scores = f1_score(most_frequent_votes, y_train[-int(validation_size):], average=None)
            f1_micro = f1_score(most_frequent_votes, y_train[-int(validation_size):], average="micro")
        else:
            scores = f1_score(most_frequent_votes, y_test, average=None)
            f1_micro = f1_score(most_frequent_votes, y_test, average="micro")

        # удаляем файлы с выходами нейронов, чтобы не было ошибок на следующей итерации
        for ensemble_number in range(1, params["n_estimators"] + 1):
            for neuron_number in range(10):
                file_path = os.path.join(path, f'neuron_{neuron_number}_from_ensemble_{ensemble_number}_validation_outputs.npy')
                if os.path.exists(file_path):
                    os.remove(file_path)

        # всегда держим веса топ конфига
        try:
            with open(os.path.join(trial_dir, 'trials.pkl'), "rb") as trials_file:
                trials = pickle.load(trials_file)
                
            best_loss = -trials.best_trial["result"]["loss"]
            if os.path.exists(os.path.join(path, "best_weights_and_features")):
                shutil.rmtree(os.path.join(path, "best_weights_and_features"))
            if f1_micro > best_loss:
                shutil.copytree(os.path.join(path, "weights_and_features"), os.path.join(path, "best_weights_and_features"))
        except:
            shutil.copytree(os.path.join(path, "weights_and_features"), os.path.join(path, "best_weights_and_features"))

        print("params:\n" + str(params) + "\nf1_micro:\n" + str(f1_micro) + "\nscores:\n" + str(scores))

        return -1 * f1_micro
    
    adjust(run, search_space, os.path.join(trial_dir, "trials.pkl"), h_evals=1, max_evals=args.ho_bound, resume=args.resume)

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

    parser.add_argument("--decode_potentials", action="store_true")
    parser.add_argument("--run_on_test_mnist", action="store_true")

    parser.add_argument("--decoding_method", type=str, default="max")
    parser.add_argument("--encoding_method", type=str, default="frequency")
    parser.add_argument("--anti_stdp", action="store_true")
    parser.add_argument("--clustering", action="store_true")
    parser.add_argument("--reversed_image", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=1)
    parser.add_argument("--every_input_spikes", action="store_true")

    parser.add_argument("--time", type=int, default=300)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    experiment(args)