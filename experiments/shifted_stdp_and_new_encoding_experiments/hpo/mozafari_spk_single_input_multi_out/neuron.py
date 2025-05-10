import sys
import argparse
import os
import pickle
import time
import logging
import numpy as np

from pathlib import Path
from sklearn.utils import shuffle

# Ensure proper import paths
_PATH_TO_REMOVE = '/s/ls4/users/romanrybka/YD/fsnn-classifiers'
if _PATH_TO_REMOVE in sys.path:
    sys.path.remove(_PATH_TO_REMOVE)
sys.path.extend([
    '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN',
    '/s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/'
])

from fsnn_classifiers.components.networks.ccn_one_spk_input_multi_spk_out import CorrelationClasswiseNetwork
from fsnn_classifiers.datasets import load_data
from neuron_base import NeuronBase


class Neuron(NeuronBase):
    """
    Encapsulates training and validation of a single-output spiking neuron in the classwise approach.
    """

    def __init__(
        self,
        neuron_idx: int,
        estimator_number: int,
        validation_size: int,
        job_id: str,
        base_dir: Path = None,
    ):
        super(Neuron, self).__init__(neuron_idx, estimator_number, validation_size, job_id, base_dir)


    @staticmethod
    def select_network_class(params: dict) -> type:
        """
        Returns the network class based on HPO params.
        """
        return CorrelationClasswiseNetwork


    def prepare_datasets(self, params: dict) -> tuple:
        """
        Loads and splits the dataset according to HPO parameters.
        """
        # Load preprocessed MNIST spikes
        spikes_base = (
            "/s/ls4/users/selibrin/spiking_researches/mozafari/"
            "datasets/MNIST_after_L2"
        )
        X_all = np.load(f"{spikes_base}/train_spikes_pooled.npy")
        y_all = np.load(f"{spikes_base}/train_targets_for_saved_spikes.npy")

        # Split validation set
        X_val = X_all[-self.validation_size:]
        y_val = y_all[-self.validation_size:]
        X_train = X_all[:-self.validation_size]
        y_train = y_all[:-self.validation_size]

        # Filter for true-class samples
        mask = y_train == int(self.neuron_idx)
        X_train = X_train[mask]
        y_train = np.zeros(mask.sum(), dtype=y_train.dtype)

        # Subsample based on max_samples
        max_samples = params.get("max_samples", 0.1)
        X_train, y_train = shuffle(X_train, y_train)
        n_sub = int(max_samples * X_train.shape[0])
        X_train = X_train[:n_sub]
        y_train = y_train[:n_sub]

        logging.info(
            f"Prepared datasets: {X_train.shape[0]} train & {X_val.shape[0]} validation samples"
        )
        return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    args = Neuron.parse_arguments()
    trainer = Neuron(
        neuron_idx=args.neuron,
        estimator_number=args.estimator_number,
        validation_size=args.validation_size,
        job_id=args.job_id,
        base_dir=Path(__file__).parent,
    )
    trainer.run()
