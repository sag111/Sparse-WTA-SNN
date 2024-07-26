import os
os.environ["PYNEST_QUIET"] = "1"

import argparse

from fsnn_classifiers.datasets import load_data
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.components.networks.correlation_classwise_network_2 import CorrelationClasswiseNetwork

from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

import numpy as np

import json
import pickle
    
def main(args):

    outputs = {"gt":[], "pred":[]}


    X_train, X_test, y_train, y_test = load_data(dataset=args.dataset, 
                                                seed=args.seed,
                                                )
    
    # we want to use cross-validation
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test])
    
    # load config

    path_to_configs = os.path.join(os.path.dirname(__file__), f'../configs/{args.dataset}/CorrelationClasswiseNetwork/')
    with open(f"{path_to_configs}/{args.plasticity}/exp_cfg_2.json", 'r') as f:
        cfg = json.load(f)

    cfg["CorrelationClasswiseNetwork"]["epochs"] = args.epochs
    cfg["CorrelationClasswiseNetwork"]["synapse_model"] = args.plasticity
    cfg["CorrelationClasswiseNetwork"]["n_estimators"] = args.n_estimators
    cfg["CorrelationClasswiseNetwork"]["max_samples"] = args.max_samples
    cfg["CorrelationClasswiseNetwork"]["V_th"] = -69.99

    if "Normalizer" in cfg.keys():
        nrm = Normalizer(norm=cfg["Normalizer"]["norm"].lower())
    elif "StandardScaler" in cfg.keys():
        nrm = StandardScaler()
    else:
        nrm = MinMaxScaler()

    grf = GRF(**cfg["GRF"])
    net = CorrelationClasswiseNetwork(**cfg["CorrelationClasswiseNetwork"])
                
    model = make_pipeline(nrm, grf, net)
                
    cv_res = cross_validate(model, 
                            X, 
                            y, 
                            cv=StratifiedKFold(n_splits=5),
                            scoring='f1_micro'
                            )['test_score']            
    return cv_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plasticity", type=str, default="stdp_nn_restr_synapse")
    parser.add_argument("--dataset", type=str, default='iris')
    parser.add_argument("--n_estimators", type=int, default=1)
    parser.add_argument("--max_samples", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    output = {key:val for key, val in vars(args).items()}
    output['f1_micro'] = list(main(args))
    output['mean'] = np.round(np.mean(output['f1_micro']),2)
    output['std'] = np.round(np.std(output['f1_micro']),2)
    print(output)
    par_token = int(args.epochs + args.seed + args.n_estimators)
    
    if args.plasticity == "stdp_nn_restr_synapse":
        pl_token = "stdp"
    elif args.plasticity == "stdp_tanh_synapse":
        pl_token = "nc"
    elif args.plasticity == "stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse":
        pl_token = "ppx"
    else:
        pl_token = args.plasticity
    
    out_path = os.path.join(os.path.dirname(__file__), args.dataset)
    os.makedirs(out_path, exist_ok=True)
    filename = os.path.join(out_path, f"ccn2_{args.dataset}_{pl_token}_{par_token}.json")
    with open(filename, 'w') as fp:
        json.dump(output, fp)

