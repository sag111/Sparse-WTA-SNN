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

from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork
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
        """ Loads and splits dataset according to HPO parameters """

        # Load dataset
        X_train, _, y_train, _ = load_data("mnist")

        # Create validation set
        X_val = X_train[-self.validation_size:]
        y_val = y_train[-self.validation_size:]

        # Remove validation set from training data
        X_train = X_train[:-self.validation_size]
        y_train = y_train[:-self.validation_size]

        # Select true class samples from dataset and prepare y_train for CorrelationClasswiseNetwork
        X_train, y_train = X_train[y_train == int(self.neuron_idx)], y_train[y_train == int(self.neuron_idx)]
        y_train = np.zeros_like(y_train)

        # Select appropriate number of training samples using `max_samples` in params
        max_samples = params.get('max_samples', 0.1)
        n_samples = int(max_samples * y_train.shape[0])
        X_train, y_train = shuffle(X_train, y_train)
        X_train, y_train = X_train[:n_samples], y_train[:n_samples]

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
