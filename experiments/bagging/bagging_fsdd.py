import os
os.environ["PYNEST_QUIET"] = "1"

import argparse

from fsnn_classifiers.datasets import load_data
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.components.networks.diehl_network import DiehlNetwork

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

import numpy as np

import json
    
def main(args):

    results = []
    for _ in range(5):
        X_train, X_test, y_train, y_test = load_data(dataset='fsdd', 
                                                    n_mfcc=args.n_mfcc,
                                                    drop_extra=True,
                                                    seed=args.seed,
                                                    )
        
        path_to_configs = os.path.join(os.path.dirname(__file__), '../configs/fsdd/BaggingDiehlNetwork/')
        with open(f"{path_to_configs}/{args.plasticity}/exp_cfg_1.json", 'r') as f:
            cfg = json.load(f)

        if "Normalizer" in cfg.keys():
            nrm = Normalizer(norm=cfg["Normalizer"]["norm"].lower())
        elif "StandardScaler" in cfg.keys():
            nrm = StandardScaler()
        else:
            nrm = MinMaxScaler()

        grf = GRF(**cfg["GRF"])
        net = DiehlNetwork(**cfg["DiehlNetwork"])
        dec = LogisticRegression(**cfg["LogisticRegression"])
                
        classifier = make_pipeline(net, dec)

        estimator = BaggingClassifier(
        base_estimator = classifier,
        bootstrap=False,
        **cfg["BaggingClassifier"],)

        model = make_pipeline(nrm, grf, estimator)
                
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)  
        res = f1_score(y_test, y_pred, average='micro')
        results.append(np.round(res, 2))

    return(results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plasticity", type=str, default="stdp_nn_symm_synapse")
    parser.add_argument("--n_mfcc", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    output = {key:val for key, val in vars(args).items()}
    output['f1_micro'] = main(args)
    print(output)
    output['mean'] = np.round(np.mean(output['f1_micro']),2)
    output['std'] = np.round(np.std(output['f1_micro']),2)

    par_token = int(args.epochs + args.seed)
    
    if args.plasticity == "stdp_nn_symm_synapse":
        pl_token = "stdp"
    elif args.plasticity == "stdp_tanh_synapse":
        pl_token = "nc"
    elif args.plasticity == "stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse":
        pl_token = "ppx"
    else:
        pl_token = args.plasticity
    
    out_path = os.path.join(os.path.dirname(__file__), 'fsdd')
    os.makedirs(out_path, exist_ok=True)
    filename = os.path.join(out_path, f"sdn_fsdd_{pl_token}_{par_token}.json")
    with open(filename, 'w') as fp:
        json.dump(output, fp)

