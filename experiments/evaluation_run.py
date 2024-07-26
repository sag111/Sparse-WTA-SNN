from fsnn_classifiers.datasets import load_data
from fsnn_classifiers.optimization.hpo import pipeline_from_params
import argparse
import os
import glob
import json
import pickle
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--plasticity", type=str)
    parser.add_argument("--cfg_name", type=str, default="exp_cfg")
    parser.add_argument("--save_weights", action="store_true")

    return parser.parse_args()

def main(args: argparse.Namespace):

    # load model config
    cfg_path = f"{os.path.dirname(__file__)}/configs/{args.dataset}/{args.model}/{args.plasticity}/{args.cfg_name}.json"
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    # parse config
    model_name = cfg.get("model_name", args.model).lower()
    
    if args.plasticity == "stdp_tanh_synapse":
        pl_name = "nc"
    elif args.plasticity == "stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse":
        pl_name = "ppx"
    else:
        pl_name = "stdp"

    exp_dir = f"{os.path.dirname(__file__)}/{model_name}/{args.dataset}/"
    os.makedirs(exp_dir, exist_ok=True)

    output_name = f"{model_name}_{args.dataset}_{pl_name}"
    result_idx = len(glob.glob(f"{exp_dir}/{output_name}_*.json"))

    X_train, X_test, y_train, y_test = load_data(dataset=args.dataset, 
                                                 n_mfcc=cfg.get("n_mfcc", 30), 
                                                 drop_extra=cfg.get("drop_extra", True),
                                                 max_train=cfg.get("max_train", 60000), 
                                                 seed=cfg.get("seed", None),
                                                 )
    
    pipe = pipeline_from_params({}, cfg)
    
    if cfg.get("cv",True) and not args.save_weights:
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        results = cross_validate(pipe, 
                                 X, 
                                 y, 
                                 cv=StratifiedKFold(n_splits=5),
                                 scoring='f1_micro'
                                 )['test_score']
        
    else:
        pipe.fit(X_train, y_train)

        if args.save_weights:
            for key, val in pipe.named_steps.items():
                if hasattr(val, "weights_"):
                    with open(f"{exp_dir}/{key}_{args.dataset}_{pl_name}_weights_{result_idx}.pkl", 'wb') as fp:
                        pickle.dump(val.weights_, fp)

        results = [f1_score(y_test, pipe.predict(X_test), average='micro')]

    mean = np.mean(results)
    std = np.std(results)

    output = cfg
    output["results"] = list(results)
    output["mean"] = mean
    output["std"] = std

    with open(f"{exp_dir}/{output_name}_{result_idx}.json", 'w') as fp:
        json.dump(output, fp)

if __name__ == "__main__":
    args = parse_args()
    main(args)