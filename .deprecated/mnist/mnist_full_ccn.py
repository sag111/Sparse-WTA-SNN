import os
os.environ["PYNEST_QUIET"] = "1"

from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork
from fsnn_classifiers.datasets.load_data import load_data

from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
import argparse
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

import pickle

import numpy as np

from tqdm import trange

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("plasticity", type=str)
    parser.add_argument("--max_train", type=int, default=15000)
    parser.add_argument("--n_estimators", type=int, default=1)
    parser.add_argument("--time", type=int, default=600)
    parser.add_argument("--test_time", type=int, default=1200)
    parser.add_argument("--max_samples", type=float, default=1.0)
    parser.add_argument("--max_features", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--decoding", type=str, default="frequency")
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
        #corr_time=0.0,
        t_ref=0.0,
        time=args.time, 
        quiet=args.quiet,
        sample_norm=1,
        #w_inh=None,
        w_init=0.0,
        #weight_normalization=None,
        early_stopping=True,
        **params,
    )

    pipe = make_pipeline(nrm, ccn)

    # load MNIST1000 data
    if args.full_ds:

        X_train, X_test, y_train, y_test = load_data("mnist", max_train=args.max_train) 
        pipe.fit(X_train, y_train)

        with open(f"{parent_directory}/results/weights_full.pkl", 'wb') as fp:
            pickle.dump(pipe.named_steps['correlationclasswisenetwork'].weights_, fp)

        data = {'sample':[], 
                'target':[], 
                'train_time':args.time, 
                'test_time':args.test_time, 
                'max_samples':args.max_samples, 
                'max_features':args.max_features, 
                'n_estimators':args.n_estimators,
                'max_train':args.max_train,
                }
             

        pipe.named_steps["correlationclasswisenetwork"].init_encoder(args.test_time)
        pipe.named_steps["correlationclasswisenetwork"].quiet = True

        n_subsets = 80
        num_samples = len(X_test) // n_subsets

        pbar = trange(n_subsets)

        for i in range(n_subsets):

            sample = X_test[i*num_samples:(i+1)*num_samples]
            target = y_test[i*num_samples:(i+1)*num_samples]
            y_pr_i = pipe.predict(sample)
            data['sample'].extend(list(np.ravel(y_pr_i)))
            data['target'].extend(list(np.ravel(target)))

            pbar.set_postfix({'f1_score':f1_score(data['sample'],data['target'],average='micro')})

            pbar.update(1)

            with open(f"{parent_directory}/results/predictions_t{args.max_train}_s{args.max_samples*100:.0f}_f{args.max_features*100:.0f}_n{args.n_estimators}.pkl", 'wb') as fp:
                pickle.dump(data, fp)

        pbar.close()
        
        return

    X_train, X_test, y_train, y_test = load_data("mnist1000")

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

        # get accuracy
        x_ts = X_train[test_idxs]
        y_ts = y_train[test_idxs]

        if args.log_results:

            x_ts_ = nrm.transform(x_ts)
            x_ts = ccn.transform(x_ts_)

            print(x_ts.shape)

            print("out freqs", x_ts.min(), x_ts.max(), x_ts.mean())

            with open(f"{parent_directory}/results/test_{args.decoding}_{i}.pkl", 'wb') as fp:
                pickle.dump((x_ts, y_ts), fp)

        y_pr = pipe.predict(x_ts)
   
        print(f"Fold {i} (ccn): {f1_score(y_ts, y_pr, average='micro')}")
   
if __name__ == "__main__":
    args = parse_args()
    run(args) 