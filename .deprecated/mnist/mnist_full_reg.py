import os
os.environ["PYNEST_QUIET"] = "1"

from sklearn.linear_model import LogisticRegression
from fsnn_classifiers.datasets.load_data import load_data

from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
import argparse
from sklearn.model_selection import StratifiedKFold

import pickle

import numpy as np

from tqdm import trange

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train", type=int, default=15000)
    parser.add_argument("--full_ds", action="store_true")
    args = parser.parse_args()
    return args

def run(args):

    parent_directory = os.path.dirname(os.path.abspath(__file__))
    
    nrm = Normalizer('l2')
    
    ccn = LogisticRegression()

    pipe = make_pipeline(nrm, ccn)

    # load MNIST1000 data
    if args.full_ds:

        X_train, X_test, y_train, y_test = load_data("mnist", max_train=args.max_train) 
        pipe.fit(X_train, y_train)

        data = {'sample':[], 
                'target':[], 
                'max_train':args.max_train,
                }
             
        n_subsets = 160
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

            with open(f"{parent_directory}/results/reg_predictions_t{args.max_train}.pkl", 'wb') as fp:
                pickle.dump(data, fp)

        pbar.close()
        
        return

    X_train, X_test, y_train, y_test = load_data("mnist1000")

    skf = StratifiedKFold(n_splits=5)
    for i, (train_idxs, test_idxs) in enumerate(skf.split(X_train, y_train)):

        x_tr = X_train[train_idxs]
        y_tr = y_train[train_idxs]

        pipe.fit(x_tr, y_tr)

        # get accuracy
        x_ts = X_train[test_idxs]
        y_ts = y_train[test_idxs]

        y_pr = pipe.predict(x_ts)
   
        print(f"Fold {i} (reg): {f1_score(y_ts, y_pr, average='micro')}")
   
if __name__ == "__main__":
    args = parse_args()
    run(args) 