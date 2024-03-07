import os
os.environ["PYNEST_QUIET"] = "1"

from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork
from fsnn_classifiers.components.decoding.own_rate_population_decoder import OwnRatePopulationDecoder
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.components.preprocessing.max_pooling import Pooling
from fsnn_classifiers.optimization.hpo import adjust

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

import numpy as np

from typing import OrderedDict

from hyperopt import hp

import argparse
import pickle

def experiment(args):

    # load MNIST1000 data
    mnist_path = f"{os.getcwd()}/fsnn_classifiers/datasets/_mnist_data/mini-mnist-1000.pickle"
    trial_dir = f"{os.getcwd()}/experiments/hpo/trials/CCN_{args.plasticity}_miniMNIST/"
    os.makedirs(trial_dir, exist_ok=True)

    with open(mnist_path, 'rb') as fp:
        data = pickle.load(fp)

    X, y = np.array(data['images']).reshape(-1, 28*28), np.array(data['labels'])
    
    search_space = OrderedDict([
        ('spike_p', hp.choice('spike_p',[0.01, 0.03, 0.05, 0.07, 0.09])),
        ('intervector_pause', hp.choice('intervector_pause', [50, 100, 150])),
        ('time', hp.choice('time', [100, 200, 300, 400, 500, 600, 700, 800, 900])),
        ('tau_s', hp.choice('tau_s', [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9])),
        #('corr_time', hp.choice('corr_time', [5., 10., 20., 30., 50.])),
        #('I_inh', hp.choice('I_inh', [-1000, -100, -10, -1])),
        #('t_ref', hp.choice('t_ref', [1., 2., 3., 4., 5.])),
        ('norm', hp.choice('norm', ['l2', 'max', 'std'])),
        ('ks', hp.choice('ks', [0, 3, 4, 5])),
        ('method', hp.choice('method', ['max', 'mean'])),
        #('n_estimators', hp.choice('n_estimators', [50, 100])),
        #('max_samples', hp.choice('max_samples', [0.5, 0.7, 0.9])),
    ])

    def run(params):

        if params["ks"] == 0:
            pool = PCA(28)
        else:
            pool = Pooling(input_shape=(28,28), 
                            kernel_size=(params["ks"],params["ks"]), 
                            stride=(params["ks"],params["ks"]), 
                            method=params["method"], 
                            pad=False)

        if params["norm"] == 'std':
            nrm = StandardScaler()
        else:
            nrm = Normalizer(norm=params['norm'])

        grf = GRF(n_fields=35)
        
        ccn = CorrelationClasswiseNetwork(
            n_fields=35,
            n_estimators=1,
            max_features=1.0,
            max_samples=1.0,
            decoding='frequency',
            synapse_model=args.plasticity,
            epochs=1,
            quiet=True,
            **params,
        )
        dec = OwnRatePopulationDecoder()

        pipe = make_pipeline(pool, 
                             nrm, 
                             grf, 
                             ccn, 
                             dec)
        pipe.fit(X, y)
        f1 = f1_score(pipe.predict(X), y, average='micro')
        print(params)
        print(f1)
        return -1 * f1
    
    adjust(run, search_space, f"{trial_dir}/trials.pkl", h_evals=1, max_evals=args.ho_bound, resume=args.resume)

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--plasticity", type=str, default="stdp_nn_pre_centered_synapse")
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--ho_bound", type=int, default=100)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    experiment(args)