import os
os.environ["PYNEST_QUIET"] = "1"

import argparse

from fsnn_classifiers.datasets import load_data
from fsnn_classifiers.components.preprocessing.gaussian_receptive_fields import GaussianReceptiveFields as GRF
from fsnn_classifiers.components.networks.diehl_network import DiehlNetwork

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import numpy as np

import json
    
def main(args):


    X_train, X_test, y_train, y_test = load_data(dataset='digits', 
                                                seed=args.seed,
                                                )
    
    # we want to use cross-validation
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test])
    
    # load config

    path_to_configs = os.path.join(os.path.dirname(__file__), '../configs/digits/BaseDiehlNetwork/')
    with open(f"{path_to_configs}/{args.plasticity}/exp_cfg.json", 'r') as f:
        cfg = json.load(f)

    cfg["DiehlNetwork"]["epochs"] = args.epochs
    cfg["DiehlNetwork"]["synapse_model"] = args.plasticity

    skf = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
    cv_res = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if "Normalizer" in cfg.keys():
            nrm = Normalizer(norm=cfg["Normalizer"]["norm"].lower())
        elif "StandardScaler" in cfg.keys():
            nrm = StandardScaler()
        else:
            nrm = MinMaxScaler()

        grf = GRF(**cfg["GRF"])
        net = DiehlNetwork(**cfg["DiehlNetwork"])
        dec = LogisticRegression(**cfg["LogisticRegression"])
                
        model = make_pipeline(nrm, grf, net, dec)
                
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        cv_res.append(np.round(f1_score(y_test, y_pred, average='micro'),2))
            
    return cv_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plasticity", type=str, default="stdp_nn_symm_synapse")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    output = {key:val for key, val in vars(args).items()}
    output['f1_micro'] = main(args)
    output['mean'] = np.round(np.mean(output['f1_micro']),2)
    output['std'] = np.round(np.std(output['f1_micro']),2)
    print(output)
    par_token = int(args.epochs + args.seed)
    
    if args.plasticity == "stdp_nn_symm_synapse":
        pl_token = "stdp"
    elif args.plasticity == "stdp_tanh_synapse":
        pl_token = "nc"
    elif args.plasticity == "stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse":
        pl_token = "ppx"
    else:
        pl_token = args.plasticity
    
    out_path = os.path.join(os.path.dirname(__file__), 'digits')
    os.makedirs(out_path, exist_ok=True)
    filename = os.path.join(out_path, f"sdn_digits_base_{pl_token}_{par_token}.json")
    with open(filename, 'w') as fp:
        json.dump(output, fp)

