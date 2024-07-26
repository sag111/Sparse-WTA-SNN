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

import nest
    
def main(args):



    with open(f"{os.path.dirname(__file__)}/samples_f1.pkl", 'rb') as fp:
        X = pickle.load(fp)

    with open(f"{os.path.dirname(__file__)}/labels_f1.pkl", 'rb') as fp:
        y = pickle.load(fp)

    with open(f"{os.path.dirname(__file__)}/weights_f1.pkl", 'rb') as fp:
        w = pickle.load(fp)
    
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
        nrm_2 = Normalizer(norm=cfg["Normalizer"]["norm"].lower())
    elif "StandardScaler" in cfg.keys():
        nrm_2 = StandardScaler()
    else:
        nrm_2 = MinMaxScaler()

    net_2 = CorrelationClasswiseNetwork(**cfg["CorrelationClasswiseNetwork"])
            
    model_2 = make_pipeline(nrm_2, net_2)
            
    model_2.fit(X, y)

    if args.test_time is not None:
        model_2.named_steps["correlationclasswisenetwork"].init_encoder(args.test_time)

    del X, y

    if "Normalizer" in cfg.keys():
        nrm_1 = Normalizer(norm=cfg["Normalizer"]["norm"].lower())
    elif "StandardScaler" in cfg.keys():
        nrm_1 = StandardScaler()
    else:
        nrm_1 = MinMaxScaler()

    net_1 = CorrelationClasswiseNetwork(**cfg["CorrelationClasswiseNetwork"])
    net_1.weights_ = w
    net_1.is_fitted_ = True
    net_1.n_features_in_ = net_2.n_features_in_
    net_1.classes_ = net_2.classes_

    model_1 = make_pipeline(nrm_1, net_1)

    X_train, _, y_train, _ = load_data(dataset="mnist", 
                                                seed=args.seed,
                                                max_train=60000,
                                                )

    y_pred_1 = model_1.predict(X_train) # first prediction
    y_pred_2 = model_2.predict(X_train) # second prediction

    print(f1_score(y_train, y_pred_1, average="micro"))
    print(f1_score(y_train, y_pred_2, average="micro"))

    samples = []
    labels = []

    for i in range(len(y_pred_1)):
        if y_pred_1[i] != y_pred_2[i]:
            samples.append(X_train[i])
            labels.append(y_train[i])

    with open(f"{os.path.dirname(__file__)}/samples_f2.pkl", 'wb') as fp:
        pickle.dump(np.array(samples), fp)

    with open(f"{os.path.dirname(__file__)}/labels_f2.pkl", 'wb') as fp:
        pickle.dump(np.array(labels), fp)

    with open(f"{os.path.dirname(__file__)}/weights_f2.pkl", 'wb') as fp:
        pickle.dump(model_2.named_steps["correlationclasswisenetwork"].weights_, fp)
            
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

