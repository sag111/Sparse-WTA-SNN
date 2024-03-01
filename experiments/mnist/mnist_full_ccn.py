import os
os.environ["PYNEST_QUIET"] = "1"

from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork
from fsnn_classifiers.components.decoding.own_rate_population_decoder import OwnRatePopulationDecoder
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.datasets.load_data import load_data

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("plasticity", type=str)
    parser.add_argument("--max_train", type=int, default=15000)
    parser.add_argument("--n_estimators", type=int, default=501)
    parser.add_argument("--n_fields", type=int, default=35)
    parser.add_argument("--max_samples", type=float, default=0.7)
    parser.add_argument("--n_components", type=int, default=28)
    parser.add_argument("--decoding", type=str, default="frequency")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    return args

def run(args):

    # load MNIST1000 data
    X_train, X_test, y_train, y_test = load_data("mnist", max_train=args.max_train)

    params = {'intervector_pause': 100, 
              'spike_p': 0.04, 
              't_ref': 4.0, 
              'tau_s': 0.5, 
              'time': 100}
    
    nrm = StandardScaler()
    grf = GRF(n_fields=args.n_fields)
    pca = PCA(n_components=args.n_components)
    ccn = CorrelationClasswiseNetwork(
        n_fields=args.n_fields,
        n_estimators=args.n_estimators,
        max_features=1.0,
        max_samples=args.max_samples,
        synapse_model=args.plasticity,
        decoding=args.decoding,
        epochs=1,
        quiet=args.quiet,
        **params,
    )
    dec = OwnRatePopulationDecoder()

    pipe = make_pipeline(pca, nrm, grf, ccn, dec)
    #pipe = make_pipeline(pca, nrm, LogisticRegression(max_iter=100000))
    pipe.fit(X_train, y_train)
    f1 = f1_score(pipe.predict(X_test), y_test, average='micro')
    print(f"Testing f1-micro: {f1:.3f} (for {args.max_train:.0f} training samples).")

if __name__ == "__main__":
    args = parse_args()
    run(args) 