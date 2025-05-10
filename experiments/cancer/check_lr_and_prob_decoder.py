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

def main():
    params = {
    'V_th': -69.9,
    'intervector_pause': 150, 
    'mu_minus': 20., #0.10559359917007424,
    'mu_plus': 20, # .4619935168191757,
    # 'norm': 'max',
    'plasticity': 'stdp_nn_restr_synapse',
    'ref_seq_interval': 9,
    'sigma_w': 0.0,
    'tau_m': 50,
    'tau_s': 0.3, 
    'time': 1000
    }

    # nrm = Normalizer('l2')
    # nrm = Normalizer('max')
    # nrm = StandardScaler()
    nrm = MinMaxScaler()

    grf = GRF(n_fields=30)

    ccn = CorrelationClasswiseNetwork(
        n_fields=30,
        n_estimators=1,
        max_features=1.0,
        max_samples=1.0,
        # synapse_model=args.plasticity,
        epochs=1,
        corr_time=0.0,
        t_ref=0.0,
        # time=args.time, 
        # quiet=args.quiet,
        sample_norm=1,
        w_inh=None,
        w_init=0.0,
        weight_normalization=None,
        early_stopping=False, #True,
        n_jobs=1,
        # job_directory=job_directory,
        # job_id=args.job_id,
        # use_entire_train_data=args.use_entire_train_data,
        **params,
    )

    pipe = make_pipeline(nrm, grf, ccn)

    X, y = load_data("fsdd", n_mfcc=40)

    skf = StratifiedKFold(n_splits=5)

    all_lr_scores = []
    all_clf_scores = []

    deltas = [0.1] + list(range(1, 50, 5))

    for j, delta_x in enumerate(deltas):
        lr_scores = []
        clf_scores = []
        
        for i, (train_idxs, test_idxs) in enumerate(skf.split(X, y)):

            x_tr = X[train_idxs]
            y_tr = y[train_idxs]


            pipe.fit(x_tr, y_tr)


        
            x_lr_tr = pipe.transform(x_tr)
            clf = ProbabilityClassifier(delta_x=delta_x, mode=0).fit(x_lr_tr, y_tr)
            lr = LogisticRegression(max_iter=10000).fit(x_lr_tr, y_tr)


            # get accuracy
            x_ts = X[test_idxs]
            y_ts = y[test_idxs]

            
            y_pr = pipe.predict(x_ts)

            x_lr_ts = pipe.transform(x_ts)
            y_pr_lr = lr.predict(x_lr_ts)
            y_pr_clf = clf.predict(x_lr_ts)
            

            fold_lr_score = f1_score(y_ts, y_pr_lr, average='micro')
            # print(f"Fold {i} (lr): {fold_lr_score}")

            fold_clf_score = f1_score(y_ts, y_pr_clf, average='micro')
            # print(f"Fold {i} (clf): {fold_clf_score}")

            lr_scores.append(fold_lr_score)
            clf_scores.append(fold_clf_score)

        all_lr_scores.append(np.mean(lr_scores))
        all_clf_scores.append(np.mean(clf_scores))
        
        print(f"all_lr_scores: {all_lr_scores}")
        print()
        print(f"deltas: {deltas[:j+1]}")
        print(f"all_clf_scores: {all_clf_scores}")



if __name__ == "__main__":
    main()