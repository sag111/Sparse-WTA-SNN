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

from random import choice
    
def main(args):

    outputs = {"gt":[], "pred":[]}


    X_train, X_test, y_train, y_test = load_data(dataset="mnist", 
                                                seed=args.seed,
                                                max_train=60000,
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

    net = CorrelationClasswiseNetwork(**cfg["CorrelationClasswiseNetwork"])
            
    model = make_pipeline(nrm, net)
            
    model.fit(X_train, y_train)

    if args.test_time is not None:
        model.named_steps["correlationclasswisenetwork"].init_encoder(args.test_time)

    y_pred = model.predict(X_train)

    samples = []
    labels = []

    for i in range(len(y_pred)):
        coin = choice([0, 1])
        if (coin == 0 and y_pred[i] != y_train[i]) or (coin == 1 and y_pred[i] == y_train[i]):
            samples.append(X_train[i])
            labels.append(y_train[i])

    with open(f"{os.path.dirname(__file__)}/samples_f1.pkl", 'wb') as fp:
        pickle.dump(np.array(samples), fp)

    with open(f"{os.path.dirname(__file__)}/labels_f1.pkl", 'wb') as fp:
        pickle.dump(np.array(labels), fp)

    with open(f"{os.path.dirname(__file__)}/weights_f1.pkl", 'wb') as fp:
        pickle.dump(model.named_steps["correlationclasswisenetwork"].weights_, fp)
            
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plasticity", type=str, default="stdp_nn_restr_synapse")
    parser.add_argument("--dataset", type=str, default='fsdd')
    parser.add_argument("--n_estimators", type=int, default=51)
    parser.add_argument("--n_mfcc", type=int, default=30)
    parser.add_argument("--drop_extra", type=bool, default=True)
    parser.add_argument("--early_stopping", type=bool, default=False)
    parser.add_argument("--max_train", type=int, default=60000)
    parser.add_argument("--test_time", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    main(args)

