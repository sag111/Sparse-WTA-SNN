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
from fsnn_classifiers.components.preprocessing.pooling import Pooling
from fsnn_classifiers.optimization.hpo import adjust
from fsnn_classifiers.datasets.load_data import load_data
from fsnn_classifiers.components.preprocessing.center_and_normalize import center_and_normalize_vec
from fsnn_classifiers.components.preprocessing.dog import DoG

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.utils import shuffle

import numpy as np

from typing import OrderedDict

from hyperopt import hp

import argparse
import pickle

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from scipy.stats import mode
import time


def run_script(neuron, ensemble_number, validation_size, job_id):
    result = subprocess.run([
        'python',
        '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/hpo/ensemble_one_neuron_grf_ccn_hpo_fold.py',
        str(neuron), 
        str(ensemble_number), 
        str(validation_size), 
        str(job_id)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Ensemble {ensemble_number} neuron {neuron} failed with return code {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        raise RuntimeError(f"Ensemble {ensemble_number} neuron {neuron} failed with return code {result.returncode}")

    return float(result.stdout.strip())


if __name__ == "__main__":

    ensemble_number = sys.argv[1]
    validation_size = sys.argv[2]
    job_id = sys.argv[3]

    n_classes = 10
    scores = 0

    path_to_params = f'/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/hpo/data_for_one_neuron/{job_id}/params.pkl'
    while not os.path.exists(path_to_params):
        pass

    # запускаем эстиматор (10 нейронов)
    with ThreadPoolExecutor(max_workers=n_classes) as executor:
        futures = [executor.submit(run_script, i, ensemble_number, validation_size, job_id) for i in range(n_classes)]
        
        for future in as_completed(futures):
            score = future.result()
            scores += score

    if scores != n_classes:
        raise Exception("scores != n_classes")

    print(scores)