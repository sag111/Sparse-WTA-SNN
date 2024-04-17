import os
os.environ["PYNEST_QUIET"] = "1"

from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork
from fsnn_classifiers.components.decoding.own_rate_population_decoder import OwnRatePopulationDecoder
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.components.preprocessing.max_pooling import Pooling
from fsnn_classifiers.datasets.load_data import load_data

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import argparse
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree

import matplotlib.pyplot as plt

import pickle

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("plasticity", type=str)
    parser.add_argument("--max_train", type=int, default=15000)
    parser.add_argument("--n_estimators", type=int, default=1)
    parser.add_argument("--time", type=int, default=600)
    parser.add_argument("--max_samples", type=float, default=1.0)
    parser.add_argument("--max_features", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--decoding", type=str, default="correlation")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log_results", action="store_true")
    parser.add_argument("--full_ds", action="store_true")
    args = parser.parse_args()
    return args

def run(args):

    parent_directory = os.path.dirname(os.path.abspath(__file__))

    
    params = {'decoding': args.decoding, 
              'V_th': -69.95, 
              'intervector_pause': 50, 
              'mu_plus': 0.5, 
              'mu_minus': 0.25,
              'ref_seq_interval': 9, 
              'sigma_w':-0.5,
              'tau_m': 40.0, 
              'tau_s': 0.4}
    
    nrm = Normalizer('l2')
    
    ccn = CorrelationClasswiseNetwork(
        n_fields=None,
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        max_samples=args.max_samples,
        synapse_model=args.plasticity,
        epochs=args.epochs,
        corr_time=0.0,
        t_ref=0.0,
        time=args.time, 
        quiet=args.quiet,
        sample_norm=1,
        w_inh=None,
        w_init=0.0,
        weight_normalization=None,
        early_stopping=True,
        **params,
    )

    #dec = DecisionTreeClassifier() 
    #pop_dec = OwnRatePopulationDecoder()
    #reg_dec = LogisticRegression(max_iter=1000000)

    pipe = make_pipeline(nrm, ccn)
    #print(X_train.max())
    #X_train[X_train > 100] = 255
    
    #X_train = np.hstack((X_train, 255 - X_train))
    #print(X_train.shape)

    # load MNIST1000 data
    if args.full_ds:

        X_train, X_test, y_train, y_test = load_data("mnist", max_train=args.max_train) 
        pipe.fit(X_train, y_train)

        with open(f"{parent_directory}/results/weights_full.pkl", 'wb') as fp:
            pickle.dump(pipe.named_steps['correlationclasswisenetwork'].weights_, fp)

        y_pr = pipe.predict(X_test)

        

        print(f"F1-micro: {f1_score(y_test, y_pr, average='micro')}")

        return

    X_train, X_test, y_train, y_test = load_data("mnist1000") # load_data("mnist", max_train=args.max_train) #

    skf = StratifiedKFold(n_splits=5)
    for i, (train_idxs, test_idxs) in enumerate(skf.split(X_train, y_train)):

        

        x_tr = X_train[train_idxs]
        y_tr = y_train[train_idxs]

        

        if args.log_results:

            x_tr = nrm.fit_transform(x_tr, y_tr)
            x_tr = ccn.fit_transform(x_tr, y_tr)
            
            with open(f"{parent_directory}/results/train_{args.decoding}_{i}.pkl", 'wb') as fp:
                pickle.dump((x_tr,y_tr), fp)

            with open(f"{parent_directory}/results/weights_{i}.pkl", 'wb') as fp:
                pickle.dump(ccn.weights_, fp)

            print("SAVED WEIGHTS")

        else:

            pipe.fit(x_tr, y_tr)
        
        #dec.fit(x_tr, y_tr)
        #pop_dec.fit(x_tr, y_tr)
        #reg_dec.fit(x_tr, y_tr)

        # get accuracy
        x_ts = X_train[test_idxs]
        y_ts = y_train[test_idxs]

        

        if args.log_results:

            x_ts_ = nrm.transform(x_ts)
            x_ts = ccn.transform(x_ts_)

            print(x_ts.shape)

            print("out freqs", x_ts.min(), x_ts.max(), x_ts.mean())

            with open(f"{parent_directory}/results/test_{args.decoding}_{i}.pkl", 'wb') as fp:
                pickle.dump((x_ts,y_ts), fp)

        else:

            y_pr = pipe.predict(x_ts)
        
        #y_pr = np.argmax(x_ts.reshape((-1, args.n_estimators, 10)), axis=-1).mean(axis=-1).round(0)
        #print(f"Fold {i} (max. freq.): {f1_score(y_ts, y_pr, average='micro')}")
        #y_pr = dec.predict(x_ts)
        #print(f"Fold {i} (tree): {f1_score(y_ts, y_pr, average='micro')}")
        #y_pr = reg_dec.predict(x_ts)
        #print(f"Fold {i} (reg): {f1_score(y_ts, y_pr, average='micro')}")
        #y_pr = pop_dec.predict(x_ts)
        #print(f"Fold {i} (pop): {f1_score(y_ts, y_pr, average='micro')}")
        #y_pr = ccn.predict(x_ts_)
        print(f"Fold {i} (ccn): {f1_score(y_ts, y_pr, average='micro')}")

        #del x_tr, y_tr, x_ts, x_ts_, y_ts, y_pr
        #del nrm, ccn, dec, pop_dec
   
if __name__ == "__main__":
    args = parse_args()
    run(args) 