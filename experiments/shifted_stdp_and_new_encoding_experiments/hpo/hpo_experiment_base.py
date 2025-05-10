import sys
path_to_remove = '/s/ls4/users/romanrybka/YD/fsnn-classifiers'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)

import argparse
import pickle
import shutil
import subprocess
import time
import os
import logging
import numpy as np

from pathlib import Path
from collections import OrderedDict
from scipy.stats import mode
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.metrics import f1_score

sys.path.append('/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/')
from fsnn_classifiers.datasets.load_data import load_data
from fsnn_classifiers.optimization.hpo import adjust


"""
_init_search_space() method must be implemented in the subclass for each experiment.
"""
class HPOExperimentBase:
    def __init__(self, args, exp_dir_name, synapse_model):
        self.args = args
        self.exp_dir_name = exp_dir_name
        self.synapse_model = synapse_model
        self.base_dir = Path(f'/s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/{self.exp_dir_name}/hpo_data/trials')
        self.trial_dir = self.base_dir / args.job_id
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.base_dir.parent / 'hpo_iteration_data' / args.job_id
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.search_space = self._init_search_space()
        self._maybe_resume()
        self.logger = logging.getLogger(__name__)
        # Save data_dir for estimator script
        with open(self.base_dir.parent.parent / 'data_dir.txt', 'w') as f:
            f.write(str(self.data_dir))

    def _init_search_space(self):
        raise NotImplementedError("_init_search_space() must be implemented in the subclass.")

    def _maybe_resume(self):
        if not self.args.dir_for_resume:
            return

        resume_dir = self.base_dir / self.args.dir_for_resume
        trials_file = resume_dir / 'trials.pkl'
        search_space_file = resume_dir / 'search_space.pkl'

        if trials_file.exists():
            shutil.copy(trials_file, self.trial_dir / 'trials.pkl')
            self.logger.info(f"Copying and continuing trials from {trials_file}.")
        else:
            self.logger.warning(f"Trials file not found at {trials_file}!")
            return

        if search_space_file.exists():
            with open(search_space_file, 'rb') as f:
                self.search_space = pickle.load(f)
            shutil.copy(search_space_file, self.trial_dir / 'search_space.pkl') 
            self.logger.info(f"Using search space from {search_space_file}")
        else:
            self.logger.warning(f"""
                Search space does not exist in: {search_space_file}!\n
                Using search space from the executed script.
            """)

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(
            description=(
                "Hyperopt hyperparameter optimization script."
            )
        )
        parser.add_argument(
            "exp_dir_name", type=Path, default=Path(__file__).parent.name,
            help="Directory name with hyperopt optimization experiment scripts."
        )
        parser.add_argument(
            "job_id", type=str,
            help="SLURM-task JOB_ID must be passed."
        )
        parser.add_argument(
            "ho_bound", type=int, default=10000,
            help="Number of iterations in the HPO Experiment."
        )
        parser.add_argument(
            "n_estimators", type=int, default=1,
            help="Size of en ensemble."
        )
        parser.add_argument(
            "dir_for_resume", default="",
            help="JOB_ID of the hyperopt experiment task to continue."
        )
        parser.add_argument(
            "validation_size", type=int, default=1000,
            help="Number of samples in the validation dataset."
        )
        
        return parser.parse_args()

    def _save_params_and_wait(self, params):
        # Save parameters for each neuron
        for e in range(1, self.args.n_estimators + 1):
            for n in range(10):
                with open(self.data_dir / f'params_{e}_{n}.pkl', 'wb') as f:
                    pickle.dump(params, f)
        # Wait for output files with potentials (or spikes)
        expected_files = np.ravel(self._expected_files(params))
        missing = {f for f in expected_files if not (self.data_dir / f).exists()}
        while missing:
            time.sleep(30)
            missing = {f for f in expected_files if not (self.data_dir / f).exists()}
        return expected_files

    def _expected_files(self, params):
        n_estimators = params.get('n_estimators', 1)
        return [
            [f'estimator_{e}_neuron_{n}_validation_outputs.npy' for n in range(10)]
            for e in range(1, n_estimators + 1)
        ]

    def _decode(self, params):
        """ Load outputs and calculate y_pred """
        ensemble_predictions = []

        expected_files = self._expected_files(params)
        for e, file_list in enumerate(expected_files, start=1):
            estimator_outputs = [np.load(self.data_dir / f) for f in file_list]
            estimator_predictions = np.hstack(estimator_outputs).argmax(axis=1)
            ensemble_predictions.append(estimator_predictions)
            np.save(self.data_dir / f"estimator_{e}_outputs.npy", estimator_outputs)

        # Convert list of predictions to array and compute majority vote
        ensemble_predictions = np.stack(ensemble_predictions, axis=0)  # Shape: (n_estimators, n_samples)
        y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=ensemble_predictions)

        return y_pred

    def _save_best_weights(self, params):
        best_weights_dir = self.data_dir / "best_weights"
        best_weights_dir.mkdir(parents=True, exist_ok=True)
        n_estimators = params.get('n_estimators', 1)
        for e in range(1, n_estimators + 1): 
            for n in range(10):
                src = self.data_dir / f"estimator_{e}_neuron_{n}_weights.npy"
                dst = best_weights_dir / f"estimator_{e}_neuron_{n}_weights.npy"
                shutil.copy(src, dst)
    
    def _clear_folder(self, folder_path, files_to_delete):
        for filename in files_to_delete:
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # delete file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # delete inner folder recursively
            except Exception as e:
                self.logger.error(f'Не удалось удалить {file_path}. Причина: {e}')
                
    def objective(self, params):
        # Saving params and waiting for output_files creation
        output_files = self._save_params_and_wait(params)
        # Loading outputs and decoding
        y_pred = self._decode(params)
        _, _, y_val, _ = load_data("mnist")
        # Last `args.validation_size` samples of train dataset are used for validation
        y_val = y_val[-self.args.validation_size:]
        score = f1_score(y_pred, y_val, average="micro")
        print(f"Params: {params}\nScore: {score:.4f}")
        # Saving best weights
        if self.best_score < score:
            self._save_best_weights(params)
            self.best_score = score
        # Delete outputs related to the current optimization iteration
        self._clear_folder(self.data_dir, output_files)
        return {'loss': -score, 'status': STATUS_OK}

    def optimize(self):
        # for saving weights of the best parameters during optimization
        self.best_score = 0

        adjust(
            self.objective, 
            self.search_space, 
            self.trial_dir / 'trials.pkl',
            h_evals=1, 
            max_evals=self.args.ho_bound, 
            resume=bool(self.args.dir_for_resume)
        )
        # The following lines are unlikely to be executed 
        # due to the limited lifetime of the task and large `max_evals` parameter in fmin().
        with open(self.trial_dir / 'best_params.pkl', 'wb') as f:
            pickle.dump(best, f)
        print(f"Best params: {best}")

def main():
    parser = argparse.ArgumentParser()

    
    args = parser.parse_args()
    exp_dir_name = ""
    synapse_model = ""

    exp = HPOExperimentBase(args, exp_dir_name, synapse_model)
    exp.optimize()

if __name__ == "__main__":
    main()
