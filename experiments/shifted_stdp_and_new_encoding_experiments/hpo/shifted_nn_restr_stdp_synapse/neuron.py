import sys
path_to_remove = '/s/ls4/users/romanrybka/YD/fsnn-classifiers'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.utils import shuffle


sys.path.append('/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN')
from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork
from fsnn_classifiers.datasets import load_data

# sys.path.append('/s/ls4/users/selibrin/tools')
# from file_creation_handler import watch_for_files

import logging
import time
import psutil


USED_SYNAPSE = "shifted_nn_restr_stdp_synapse"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training and Validation of a single output (class) spiking neuron in a classwise approach.")
    parser.add_argument("neuron", type=int, help="Neuron index that determines his class as `neuron_idx` % `n_classes`.")
    parser.add_argument("ensemble_number", type=int, help="Ensemble number that may specify the training data.")
    parser.add_argument("validation_size", type=int, help="Number of samples to use for validation.")
    parser.add_argument("job_id", type=str, help="`JOB_ID` of a SLURM-task that specifies directory for data.")
    # parser.add_argument("n_jobs", type=int, help="Number of available cores for parallel computing.")
    return parser.parse_args()


def load_hpo_params(job_data_path: Path, neuron_idx: int, ensemble_num: int,) -> dict:
    """
    Waits for appropriate file with hyperparametes, loads it and deletes the file.
    """
    params_file = job_data_path / f"params_{ensemble_num}_{neuron_idx}.pkl"

    # if not params_file.exists():
    #     watch_for_files(str(job_data_path), {params_file.name})

    while not params_file.exists():
        time.sleep(5)

    with params_file.open("rb") as f:
        params = pickle.load(f)
    params_file.unlink()

    return params


def select_network_class(params: dict) -> type:
    """ 
        Returns the appropriate network model based on params.
        Right now nothing to select. 
    """
    return CorrelationClasswiseNetwork


def prepare_datasets(
    params: dict,
    neuron_idx: int,
    ensemble_num: int,
    validation_size: int,
    base_path: Path,
):
    """ Loads and splits dataset according to HPO parameters """

    # Load dataset
    X_train, _, y_train, _ = load_data("mnist")

    # Create validation set
    X_validation = X_train[-validation_size:]
    y_validation = y_train[-validation_size:]

    # Remove validation set from training data
    X_train = X_train[:-validation_size]
    y_train = y_train[:-validation_size]

    # Select true class samples from dataset and prepare y_train for CorrelationClasswiseNetwork
    X_train, y_train = X_train[y_train == int(neuron_idx)], y_train[y_train == int(neuron_idx)]
    y_train = np.zeros_like(y_train)

    # Select appropriate number of training samples using `max_samples` in params
    max_samples =  params.get('max_samples', 0.1)
    X_train, y_train = shuffle(X_train, y_train)
    X_train, y_train = X_train[:int(max_samples*y_train.shape[0])], y_train[:int(max_samples*y_train.shape[0])]

    return X_train, X_validation, y_train, y_validation


def main():
    args = parse_arguments()
    base_path = Path(f"/s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/{USED_SYNAPSE}")
    job_data_path = base_path / f"hpo_data/hpo_iteration_data/{args.job_id}"

    # Logging config
    job_id = os.environ.get("SLURM_JOB_ID") + '_' + os.environ.get("SLURM_ARRAY_TASK_ID")
    log_path = base_path / f"hpo_data/logs/{job_id}/{job_id}_{args.ensemble_number}_{args.neuron}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    while True:
        p = psutil.Process()
        logging.info(f'Available CPUs: {p.cpu_affinity()}')
        logging.info(f'Waiting for file with parameters: {job_data_path / f"params_{args.ensemble_number}_{args.neuron}.pkl"}')
        
        # Get hyperparameters
        params = load_hpo_params(job_data_path, args.neuron, args.ensemble_number)

        # Choose network model
        NetworkModel = select_network_class(params)
        params["class_name"] = args.neuron

        logging.info('Preparing datasets ...')
        logging.info(f'`max_samples` = {params.get("max_samples", 0.1)}')
        # Prepare datasets
        X_train, X_validation, y_train, y_validation = prepare_datasets(
            params,
            neuron_idx=args.neuron,
            ensemble_num=args.ensemble_number,
            validation_size=args.validation_size,
            base_path=job_data_path
        )

        # Clean up params
        for key in ["n_estimators", "max_features", "max_samples"]:
            params.pop(key, None)

        logging.info('Model initialization ...')
        # Instantiate the network
        model = NetworkModel(
            n_estimators=1,
            max_features=1.0,
            max_samples=1.0,
            corr_time=0.0,
            quiet=True,
            sample_norm=1,
            w_inh=None,
            weight_normalization=None,
            early_stopping=False,
            n_jobs=1,
            job_id=args.job_id,
            ensemble_number=args.ensemble_number,
            **params,
        )

        logging.info('Model fitting ...')
        # Fit the model
        model.fit(X_train, y_train)
        # Saving weights
        weights = model.weights_
        np.save(job_data_path / f"neuron_{args.neuron}_weights.npy", weights)

        logging.info('Model eval ...')
        # Transform and save outputs
        val_out = model.transform(X_validation)
        np.save(
            job_data_path / f"neuron_{args.neuron}_ensemble_{args.ensemble_number}_validation_outputs.npy",
            val_out
        )


if __name__ == "__main__":
    main()
