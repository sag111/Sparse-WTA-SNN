import os
os.environ["PYNEST_QUIET"] = "1"

import argparse

from collections import OrderedDict
from hyperopt import hp

from fsnn_classifiers.datasets import load_data
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork

from fsnn_classifiers.optimization.hpo import adjust

from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold

import numpy as np


def main(args):

    exp_dir = f"{os.getcwd()}/experiments/hpo/trials/CCN_{args.plasticity}_{args.dataset}_mean_v2/"
    os.makedirs(exp_dir, exist_ok=True)
    
    if 'diehl' in args.dataset.split('-'):
        grf_space = [None]
    else:
        grf_space = [10, 15, 20, 25, 30]
    
    space = OrderedDict(
            [
                ('CCN+V_th', hp.choice('CCN+V_th', [-69.6, -69.7, -69.8, -69.9, -69.95])),
                ('CCN+tau_s', hp.choice('CCN+tau_s', [0.1, 0.3, 0.5, 0.7, 0.9])),
                ('CCN+tau_m', hp.choice('CCN+tau_m', [10.0, 30.0, 50.0, 70.0, 90.0])),
                ('CCN+ref_seq_interval', hp.choice('CCN+ref_seq_interval', [3,5,7,9])),
                ('CCN+intervector_pause', hp.choice('CCN+intervector_pause', [50,100,150])),
                ('CCN+sigma_w', hp.choice('CCN+sigma_w', [-0.5, 0.0, 0.5])),
                ('CCN+max_samples', hp.choice('CCN+max_samples', [0.1, 0.3, 0.5, 0.7])),
                ('CCN+max_features', hp.choice('CCN+max_features', [0.3, 0.5, 0.7, 0.9, 1.0])),
                ('GRF+n_fields', hp.choice('GRF+n_fields', grf_space)),
                ('PRP+norm', hp.choice('PRP+norm', ['L2','MMS'])),
            ]
        )
    
    pipeline_component_names = ('PRP', 'GRF', 'CCN')
    modules = ('PRP', GRF, CorrelationClasswiseNetwork)
    
    def run(params):

        print(params) 

        X_train, X_test, y_train, y_test = load_data(dataset=args.dataset, 
                                                 n_mfcc=args.n_mfcc, 
                                                 drop_extra=args.drop_extra, 
                                                 seed=args.seed,
                                                 )
        
        

        class_means = {i: np.mean(X_train[y_train != i]) for i in np.unique(y_train)}

        # subtract the landscape
        for i in range(len(X_train)):
            X_train[i] = X_train[i] - class_means[y_train[i]]
        
        if args.dataset in ['iris', 'cancer', 'mnist1000']:
            raise ValueError(f"{args.dataset} is not supported")
        
        pipeline_components = []
        for module_name, module_ in zip(pipeline_component_names, modules):
            if module_name == "CCN":
                module_params = {"n_estimators":11,
                                 "time":1000,
                                 "t_ref":0.0,
                                 "w_init":0.0,
                                 "mu_plus":0.5,
                                 "mu_minus":0.25,
                                 "n_fields":params["GRF+n_fields"],
                                 "synapse_model":args.plasticity, 
                                 "bootstrap_features":False,
                                 "early_stopping":False, 
                                 "quiet":True}
            else:
                module_params = dict() 
            for param_name, param_ in params.items():  
                if param_name.split("+")[0] == module_name:
                    if module_ == "PRP":
                        if param_ in ["L2", "L1", "MAX"]:
                            module = Normalizer
                            param = param_.lower()
                        elif param_ == "MMS":
                            module = MinMaxScaler
                            param = "skip"
                        elif param_ == "SS":
                            module = StandardScaler
                            param = "skip"
                    else:
                        module = module_
                        param = param_
                    
                    if param != "skip":
                        module_params[param_name.split("+")[-1]] = param
    
            pipeline_components.append((module_name, module(**module_params)))
               
        model = Pipeline(steps=pipeline_components)

        model.fit(X_train, y_train)

        y_tr_pred = model.predict(X_train)
        y_ts_pred = np.empty((X_test.shape[0], len(class_means)))
        for i in range(len(X_test)):
            for c, mu in class_means.items():
                x_ts = X_test[i] - mu
                x_ts = model.transform(x_ts.reshape(1,-1)) # shape = (1, n_estimators, n_classes)
                # we want only max frequency for class <c>
                y_ts_pred[i] = np.max(x_ts[:,:,c].reshape(1, -1), axis=-1)
        
        y_ts_pred = np.argmax(y_ts_pred, axis=-1)

    
        result = f1_score(y_train, y_tr_pred, average='micro') * f1_score(y_test, y_ts_pred, average='micro')
    
        print(f1_score(y_test, y_ts_pred, average='micro'))    
        return -1 * result
               
    best_space = adjust(run, space, f"{exp_dir}/trial.pkl", h_evals=1, max_evals=args.ho_bound, resume=args.resume)
    print(best_space)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=str, default="CorrelationClasswiseNetwork")
    parser.add_argument("--dataset", type=str, default="fsdd")
    parser.add_argument("--plasticity", type=str, default="stdp_nn_restr_synapse")
    parser.add_argument("--n-mfcc", type=int, default=30)
    parser.add_argument("--drop-extra", type=bool, default=False)
    parser.add_argument("--ho-bound", type=int, default=50)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--seed", default=None)
    args = parser.parse_args()
    main(args)



