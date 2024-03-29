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
    #parser.add_argument("--max_train", type=int, default=15000)
    parser.add_argument("--n_estimators", type=int, default=1)
    parser.add_argument("--time", type=int, default=600)
    parser.add_argument("--max_samples", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--decoding", type=str, default="correlation")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log_results", action="store_true")
    args = parser.parse_args()
    return args

def run(args):

    parent_directory = os.path.dirname(os.path.abspath(__file__))

    # load MNIST1000 data
    X_train, X_test, y_train, y_test = load_data("mnist1000") #load_data("mnist", max_train=args.max_train) #
    params = {'decoding': args.decoding, 
              'intervector_pause': 150, 
              'ref_seq_interval': 21, 
              't_ref': 5.0, 
              'tau_s': 0.4}
    
    #X_train = 255 - X_train

    skf = StratifiedKFold(n_splits=5)
    for i, (train_idxs, test_idxs) in enumerate(skf.split(X_train, y_train)):

        nrm = Normalizer('l2')
    
        ccn = CorrelationClasswiseNetwork(
            n_fields=None,
            n_estimators=args.n_estimators,
            max_features=1.0,
            max_samples=args.max_samples,
            synapse_model=args.plasticity,
            epochs=args.epochs,
            corr_time=20.0,
            time=args.time, 
            quiet=args.quiet,
            sample_norm=1,
            w_inh=None,
            weight_normalization=None,
            early_stopping=True,
            **params,
        )

        dec = DecisionTreeClassifier() 
        pop_dec = OwnRatePopulationDecoder()
        reg_dec = LogisticRegression(max_iter=1000000)

        x_tr = X_train[train_idxs]
        y_tr = y_train[train_idxs]

        x_tr = nrm.fit_transform(x_tr, y_tr)
        x_tr = ccn.fit_transform(x_tr, y_tr)

        #weights = ccn.weights_
        #new_weights = []
        #for pre_idx, pos_idx, weight in weights:
        #    new_weights.append((pre_idx, pos_idx, 1.0-weight))
        #ccn.weights_ = new_weights

        if args.log_results:
            with open(f"{parent_directory}/results/train_{args.decoding}_{i}.pkl", 'wb') as fp:
                pickle.dump((x_tr,y_tr), fp)

            with open(f"{parent_directory}/results/weights_{i}.pkl", 'wb') as fp:
                pickle.dump(ccn.weights_, fp)
        
        dec.fit(x_tr, y_tr)
        pop_dec.fit(x_tr, y_tr)
        reg_dec.fit(x_tr, y_tr)

        if args.log_results:

            # Step 3: Save the resulting figure
            plt.figure(figsize=(60,30), dpi=300)  # set figsize to control the size of the plot
            plot_tree(dec, filled=True, rounded=True)
            plt.title("Decision Tree")
            plt.savefig(f"{parent_directory}/results/decision_tree_{args.decoding}_{i}.png")  # save the figure

        # get accuracy
        x_ts = X_train[test_idxs]
        y_ts = y_train[test_idxs]

        x_ts_ = nrm.transform(x_ts)
        x_ts = ccn.transform(x_ts_)

        if args.log_results:
            with open(f"{parent_directory}/results/test_{args.decoding}_{i}.pkl", 'wb') as fp:
                pickle.dump((x_ts,y_ts), fp)
        
        y_pr = np.argmax(x_ts, axis=-1)
        print(f"Fold {i} (max. freq.): {f1_score(y_ts, y_pr, average='micro')}")
        y_pr = dec.predict(x_ts)
        print(f"Fold {i} (tree): {f1_score(y_ts, y_pr, average='micro')}")
        y_pr = reg_dec.predict(x_ts)
        print(f"Fold {i} (reg): {f1_score(y_ts, y_pr, average='micro')}")
        y_pr = pop_dec.predict(x_ts)
        print(f"Fold {i} (pop): {f1_score(y_ts, y_pr, average='micro')}")
        y_pr = ccn.predict(x_ts_)
        print(f"Fold {i} (ccn): {f1_score(y_ts, y_pr, average='micro')}")

        del x_tr, y_tr, x_ts, x_ts_, y_ts, y_pr
        del nrm, ccn, dec, pop_dec
   
if __name__ == "__main__":
    args = parse_args()
    run(args) 