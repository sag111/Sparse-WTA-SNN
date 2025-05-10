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
sys.path.append(
    '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN'
)
from fsnn_classifiers.datasets import load_data


class NeuronBase:
    """
    Encapsulates training and validation of a single-output spiking neuron in the classwise approach.
    """

    def __init__(
        self,
        neuron_idx: int,
        estimator_number: int,
        validation_size: int,
        job_id: str,
        base_dir: Path,
    ):
        self.neuron_idx = neuron_idx
        self.estimator_number = estimator_number
        self.validation_size = validation_size
        self.job_id = job_id
        self.base_path = base_dir

        # Paths for HPO data and logs
        self.job_data_path = self.base_path / f"hpo_data/hpo_iteration_data/{self.job_id}"
        slurm_job = os.environ.get("SLURM_JOB_ID", "")
        slurm_array = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        composite_id = f"{slurm_job}_{slurm_array}" if slurm_job and slurm_array else self.job_id
        log_dir = self.base_path / f"hpo_data/logs/{composite_id}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{composite_id}_{self.estimator_number}_{self.neuron_idx}.log"

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Initialized Neuron trainer")

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(
            description=(
                "Training and Validation of a single-output (class) spiking neuron "
                "in a classwise approach."
            )
        )
        parser.add_argument(
            "neuron", type=int,
            help="Neuron index that determines its class as `neuron_idx` % `n_classes`."
        )
        parser.add_argument(
            "estimator_number", type=int,
            help="Estimator number that may specify the training data."
        )
        parser.add_argument(
            "validation_size", type=int,
            help="Number of samples to use for validation."
        )
        parser.add_argument(
            "job_id", type=str,
            help="`JOB_ID` of a SLURM-task that specifies directory for data."
        )
        return parser.parse_args()

    def load_hpo_params(self) -> dict:
        """
        Wait for the HPO parameters file, load its content, then delete the file.
        """
        params_file = self.job_data_path / f"params_{self.estimator_number}_{self.neuron_idx}.pkl"
        logging.info(f"Waiting for parameters file: {params_file}")
        while not params_file.exists():
            time.sleep(5)

        with params_file.open("rb") as f:
            params = pickle.load(f)
        params_file.unlink()
        logging.info("Loaded HPO parameters")
        return params

    @staticmethod
    def select_network_class(params: dict) -> type:
        """
        Returns the network class based on HPO params.
        """
        raise NotImplementedError("select_network_class() must be implemented in the subclass.")

    def prepare_datasets(self, params: dict) -> tuple:
        """
        Loads and splits the dataset according to HPO parameters.
        """
        raise NotImplementedError("prepare_datasets() must be implemented in the subclass.")

    def run(self):
        """
        Main loop: load params, prepare data, train model, and save outputs.
        """
        while True:
            params = self.load_hpo_params()
            NetworkModel = self.select_network_class(params)
            params["class_name"] = self.neuron_idx

            logging.info("Preparing datasets...")
            X_train, X_val, y_train, y_val = self.prepare_datasets(params)

            # Clean unused HPO keys
            for key in ("n_estimators", "max_features", "max_samples",):
                params.pop(key, None)

            logging.info("Initializing model...")
            model = NetworkModel(
                n_estimators=1,
                max_features=1.0,
                max_samples=1.0,
                quiet=True,
                sample_norm=1,
                w_inh=None,
                weight_normalization=None,
                early_stopping=False,
                n_jobs=1,
                job_id=self.job_id,
                **params,
            )

            logging.info("Fitting model...")
            model.fit(X_train, y_train)

            # Save weights
            weights = model.weights_
            w_path = (
                self.job_data_path /
                f"estimator_{self.estimator_number}_neuron_{self.neuron_idx}_weights.npy"
            )
            np.save(w_path, weights)
            logging.info(f"Saved weights to {w_path}")

            # Evaluate on validation data
            logging.info("Evaluating model...")
            val_out = model.transform(X_val)
            out_path = (
                self.job_data_path /
                f"estimator_{self.estimator_number}_neuron_{self.neuron_idx}_validation_outputs.npy"
            )
            np.save(out_path, val_out)
            logging.info(f"Saved validation outputs to {out_path}")


if __name__ == "__main__":
    args = NeuronBase.parse_arguments()
    trainer = NeuronBase(
        neuron_idx=args.neuron,
        estimator_number=args.estimator_number,
        validation_size=args.validation_size,
        job_id=args.job_id,
    )
    trainer.run()
