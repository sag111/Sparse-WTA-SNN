import os
os.environ["PYNEST_QUIET"] = "1"

from fsnn_classifiers.datasets import load_data
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.components.networks.correlation_classwise_network_2 import CorrelationClasswiseNetwork

from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold

X_train, X_test, y_train, y_test = load_data(dataset="mnist1000",  
                                            seed=1337,
                                            )

module_params = {"n_estimators":1,
                "max_samples":1.0,
                "max_features":1.0,
                "time":1000,
                "t_ref":0.0,
                "w_init":0.0,
                "synapse_model":"stdp_nn_restr_synapse", 
                "bootstrap_features":False,
                "early_stopping":False, 
                "n_fields":None,
                "quiet":False}

hpo_params = {'CCN+mu_minus': 0.0, 'CCN+mu_plus': 0.0, 'CCN+rate': 2000, 'CCN+sigma_w': 0.0, 'CCN+tau_minus': 5.0, 'CCN+tau_plus': 50.0}

params = {key.split('+')[-1]:val for key, val in hpo_params.items()}
for key, val in module_params.items():
    params[key] = val

nrm = Normalizer(norm='max')
ccn = CorrelationClasswiseNetwork(**params)

pipe = make_pipeline(nrm, ccn)

pipe.fit(X_train, y_train)

print(f1_score(y_test, pipe.predict(X_test), average='micro'))

