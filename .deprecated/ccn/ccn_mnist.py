import os
os.environ["PYNEST_QUIET"] = "1"

import argparse

from fsnn_classifiers.datasets import load_data
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork

from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import numpy as np

import json
import pickle

from typing import Optional
    
def main(args):

    outputs = {"gt":[], "pred":[]}


    X_train, X_test, y_train, y_test = load_data(dataset=args.dataset, 
                                                seed=args.seed,
                                                drop_extra=args.drop_extra,
                                                n_mfcc=args.n_mfcc,
                                                max_train=args.max_train,
                                                )
    
    # load config

    path_to_configs = os.path.join(os.path.dirname(__file__), f'../configs/{args.dataset}/CorrelationClasswiseNetwork/')
    with open(f"{path_to_configs}/{args.plasticity}/exp_cfg.json", 'r') as f:
        cfg = json.load(f)

    cfg["CorrelationClasswiseNetwork"]["epochs"] = args.epochs
    cfg["CorrelationClasswiseNetwork"]["synapse_model"] = args.plasticity
    cfg["CorrelationClasswiseNetwork"]["learning_rate"] = args.learning_rate
    cfg["CorrelationClasswiseNetwork"]["n_estimators"] = args.n_estimators
    cfg["CorrelationClasswiseNetwork"]["early_stopping"] = args.early_stopping

    
    cv_res = []

    if "Normalizer" in cfg.keys():
        nrm = Normalizer(norm=cfg["Normalizer"]["norm"].lower())
    elif "StandardScaler" in cfg.keys():
        nrm = StandardScaler()
    else:
        nrm = MinMaxScaler()

    grf = GRF(**cfg["GRF"])
    net = CorrelationClasswiseNetwork(**cfg["CorrelationClasswiseNetwork"])
            
    model = make_pipeline(nrm, grf, net)
            
    model.fit(X_train, y_train)

    if args.test_time is not None:
        model.named_steps["correlationclasswisenetwork"].init_encoder(args.test_time)

    y_pred = model.predict(X_test)

    outputs["gt"].append(y_test)
    outputs["pred"].append(y_pred)

    cv_res.append(np.round(f1_score(y_test, y_pred, average='micro'),2))

    print(f"Outputs saved to: {os.path.dirname(__file__)}/{args.dataset}_{args.plasticity}_outputs.pkl")

    with open(f"{os.path.dirname(__file__)}/{args.dataset}_{args.plasticity}_outputs.pkl", 'wb') as f:
        pickle.dump(outputs, f)
            
    return cv_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plasticity", type=str, default="stdp_nn_restr_synapse")
    parser.add_argument("--dataset", type=str, default='fsdd')
    parser.add_argument("--n_estimators", type=int, default=101)
    parser.add_argument("--n_mfcc", type=int, default=30)
    parser.add_argument("--drop_extra", type=bool, default=True)
    parser.add_argument("--early_stopping", type=bool, default=False)
    parser.add_argument("--max_train", type=int, default=60000)
    parser.add_argument("--test_time", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    output = {key:val for key, val in vars(args).items()}
    output['f1_micro'] = main(args)
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
    filename = os.path.join(out_path, f"ccn_{args.dataset}_{pl_token}_{par_token}.json")
    with open(filename, 'w') as fp:
        json.dump(output, fp)

