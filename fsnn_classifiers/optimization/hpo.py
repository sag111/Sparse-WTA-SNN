
from hyperopt import hp, fmin, tpe, space_eval, Trials
import pickle
from collections import OrderedDict
from fsnn_classifiers.datasets import load_data
import numpy as np
import inspect
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score
from sklearn.ensemble import BaggingClassifier

def underscore_to_camel(name):
    parts = [item.capitalize() for item in name.split('_')]
    return "".join(parts)

def function_args_from_cfg(fun: callable, config: dict) -> dict:
    return {
        key: val
        for key, val in config.items()
        if key in inspect.getfullargspec(fun).args
    }

def dynamic_import(components):
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def run_evals(run, space, n_evals, trial_name):
    trials = pickle.load(open(trial_name, "rb"))
    best = fmin(run, space, algo=tpe.suggest, trials=trials, max_evals=n_evals)
    pickle.dump(trials, open(trial_name, "wb"))
    return best


def adjust(run, space, trial_name, h_evals=10, max_evals=100, resume=False):
    if not resume:
        trials = Trials()
        pickle.dump(trials, open(trial_name, "wb"))
    else:
        trials = pickle.load(open(trial_name, "rb")) 

    n_evals = h_evals
    max_evals = max_evals
    while n_evals <= max_evals:
        best = run_evals(run, space, n_evals, trial_name)
        best_space = space_eval(space, best)
        n_evals += h_evals

    best_space = space_eval(space, best)
    print(best_space)

    return best_space

def space_from_cfg(cfg):

    od_args = [
        (key, hp.choice(key, val))
        for key, val in cfg.items()
    ]

    return OrderedDict(od_args)


def pipeline_from_params(params: dict, cfg: dict):

    module_names = [["fsnn_classifiers", "components"] + item.split('.') for item in cfg["module_names"]]
    for i in range(len(module_names)):
        module_names[i].append(underscore_to_camel(module_names[i][-1]))
    model_classes = [
        dynamic_import(module_name)
        for module_name in module_names # e.g, module_name = ["components", "networks", "correlation_classwise_network"]
    ]

    if "BaggingClassifier" in cfg.keys():
        pipeline_items = []
        cls_items = []
        for module_name, model_class in zip(module_names, model_classes):
            cfg_args = function_args_from_cfg(model_class.__init__, cfg[module_name[-1]]) if module_name[-1] in cfg.keys() else {}
            param_args = function_args_from_cfg(model_class.__init__, params)
            for key, val in param_args.items():
                cfg_args[key] = val
            model = model_class(**cfg_args)
            if module_name[2] == "preprocessing":
                pipeline_items.append((module_name[-1], model))
            else:
                cls_items.append((module_name[-1], model))
        base_classifier = Pipeline(steps=cls_items) # this part is inside the bagging classifier
        classifier = BaggingClassifier(
            estimator=base_classifier,
            **cfg["BaggingClassifier"]
        )

        pipeline_items.append(('BaggingClassifier', classifier))
    else:  
        pipeline_items = []
        for module_name, model_class in zip(module_names, model_classes):
            cfg_args = function_args_from_cfg(model_class.__init__, cfg[module_name[-1]]) if module_name[-1] in cfg.keys() else {}
            param_args = function_args_from_cfg(model_class.__init__, params)
            for key, val in param_args.items():
                cfg_args[key] = val
            model = model_class(**cfg_args)
            pipeline_items.append((module_name[-1], model))

    return Pipeline(steps=pipeline_items)


def experiment_run(params: dict, cfg: dict) -> float:

    print(params)

    X_train, X_test, y_train, y_test = load_data(dataset=cfg["dataset"], 
                                                 n_mfcc=cfg.get("n_mfcc", 30), 
                                                 max_train=cfg.get("max_train", 60000),
                                                 drop_extra=cfg.get("drop_extra", True), 
                                                 seed=cfg.get("seed", None),
                                                 )
    
    pipe = pipeline_from_params(params, cfg)
    
    if cfg.get("cv",True):
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        results = cross_validate(pipe, 
                                 X, 
                                 y, 
                                 cv=StratifiedKFold(n_splits=5),
                                 scoring='f1_micro'
                                 )['test_score']
        print(np.mean(results))
        return -1 * (np.min(results) - np.std(results)) # pessimistic approach
    else:
        pipe.fit(X_train, y_train)

        train_score = f1_score(y_train, pipe.predict(X_train), average='micro')
        test_score = f1_score(y_test, pipe.predict(X_test), average='micro')

        print(test_score)
        return -1 * train_score * test_score

