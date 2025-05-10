import sys 

sys.path.insert(0, '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN')

from fsnn_classifiers.components.decoding.probability_decoder import ProbabilityClassifier
from fsnn_classifiers.datasets.load_data import load_data

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score


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
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree

import matplotlib.pyplot as plt

import pickle

import numpy as np

from fsnn_classifiers.components.decoding.probability_decoder import ProbabilityClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from fsnn_classifiers.datasets.load_data import load_data
from sklearn.metrics import f1_score

import os


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
    args = parser.parse_args()
    return args

def main(args):

    parent_directory = os.path.dirname(os.path.abspath(__file__))
    print(parent_directory)

    # Полный путь к директории для job_id
    job_directory = os.path.join(parent_directory, 'results', args.job_id)

    # Создание всех вложенных директорий, если они не существуют
    os.makedirs(job_directory, exist_ok=True)

    params = {
        'V_th': -68, 
        'norm': 'std', 
        'ref_seq_interval': 75, 
        'sigma_w': 0.5, 
        'tau_m': 70, 
        'tau_s': 11.0, 
        'w_init': 0.07961
    }

    # nrm = Normalizer('l2')
    # nrm = Normalizer('max')
    nrm = StandardScaler()
    # nrm = MinMaxScaler()

    grf = GRF(n_fields=30)

    ccn = CorrelationClasswiseNetwork(
        n_fields=30,
        n_estimators=1,
        max_features=1.0,
        max_samples=1.0,
        synapse_model="stdp_tanh_synapse", #args.plasticity,
        epochs=1,
        corr_time=0.0,
        t_ref=0.0,
        time=1000, 
        # quiet=args.quiet,
        sample_norm=1,
        w_inh=None,
        # w_init=0.0,
        weight_normalization=None,
        early_stopping=False, #True,
        n_jobs=1,
        job_directory=job_directory,
        job_id=args.job_id,
        record_weights=True,
        # use_entire_train_data=args.use_entire_train_data,
        **params,
    )

    pipe = make_pipeline(grf, ccn)

    X_train, X_test, y_train, y_test = load_data("fsdd", n_mfcc=40)

    X_train = np.clip(nrm.fit_transform(X_train), a_min=0, a_max=None)
    X_test = np.clip(nrm.transform(X_test), a_min=0, a_max=None)


    # all_lr_scores = []
    # all_clf_scores = []
    # all_scores = []

    # deltas = [0.1] + list(range(1, 110, 5))

    # for j, delta_x in enumerate(deltas):

    pipe.fit(X_train, y_train)
        
    x_lr_tr = pipe.transform(X_train)

    # clf = ProbabilityClassifier(delta_x=delta_x, mode=0).fit(x_lr_tr, y_train)
    lr = LogisticRegression(max_iter=10000).fit(x_lr_tr, y_train)

    x_lr_ts = pipe.transform(X_test)
        
    y_pr = pipe.predict(X_test)
    y_pr_lr = lr.predict(x_lr_ts)
    # y_pr_clf = clf.predict(x_lr_ts)
        
    score = f1_score(y_test, y_pr, average='micro')
    lr_score = f1_score(y_test, y_pr_lr, average='micro')
    # clf_score = f1_score(y_test, y_pr_clf, average='micro')

    # all_lr_scores.append(lr_score)
    # all_clf_scores.append(clf_score)
    # all_scores.append(score)
        
    print(f"ccn score: {score}")
    print(f"lr score: {lr_score}")


if __name__ == "__main__":
    args = parse_args()
    main(args) 