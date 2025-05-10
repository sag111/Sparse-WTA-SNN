import sys
path_to_remove = '/s/ls4/users/romanrybka/YD/fsnn-classifiers'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)

import argparse
import pickle
import shutil
import subprocess
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
from scipy.stats import mode
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.metrics import f1_score

import os

sys.path.append('/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/')
from fsnn_classifiers.datasets.load_data import load_data
from fsnn_classifiers.optimization.hpo import adjust
# from fsnn_classifiers.components.decoding.mahalanobis_distance_decoder import MahalanobisDecoder

# sys.path.append('/s/ls4/users/selibrin/tools')
# from file_creation_handler import watch_for_files

import logging


logger = logging.getLogger(__name__)
USED_SYNAPSE = "shifted_nn_pre_centered_stdp_synapse"


def clear_folder(folder_path, files_to_delete):
    for filename in files_to_delete:
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # delete file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # delete inner folder recursively
        except Exception as e:
            print(f'Не удалось удалить {file_path}. Причина: {e}')


class HPOExperiment:
    def __init__(self, args):
        self.args = args
        self.base_dir = Path(f'/s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/{USED_SYNAPSE}/hpo_data/trials')
        self.trial_dir = self.base_dir / args.job_id
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.base_dir.parent / 'hpo_iteration_data' / args.job_id
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.search_space = self._init_search_space()
        self._maybe_resume()
        # Save data_dir for estimator script
        with open('data_dir.txt', 'w') as f:
            f.write(str(self.data_dir))

    def _init_search_space(self):
        ss = OrderedDict([
            ('encoding_method', self.args.encoding_method), # 'frequency', 'time'
            ('decoding_method', self.args.decoding_method), # 'max', 'mahalanobis'
            ('decode_potentials', True), 
            ('V_th', 10000),
            ('synapse_model', USED_SYNAPSE),
            ('neuron_model', 'iaf_psc_exp'), # iaf_cond_exp_adaptive
            ('resolution', 0.1),
            ('tau_s', 0.),
            ('t_ref', 0.),
            ('sigma_w', None),
            ('tau_m', 1e7),
            ('w_init', 0.),

            ('time', 300),
            ('ref_seq_interval', hp.randint('ref_seq_interval', 2, 11)),

            ('tau_minus', hp.choice('tau_minus', np.arange(1., 301, 20.))),
            ('tau_plus', hp.choice('tau_plus', np.arange(1., 301, 20.))),
            ('mu_plus', hp.choice('mu_plus', np.arange(0.01, 1.01, 0.05))),
            ('mu_minus', hp.choice('mu_minus', np.arange(0.01, 1.01, 0.05))),
            ('alpha', hp.choice('alpha', [hp.uniform('alpha_small', 0., 10.), hp.loguniform('alpha_big', 2., 8.)])),
            ('lr', hp.uniform('lr', 1e-6, 1)),
            ('shift_minus', hp.uniform('shift_minus', 0., 0.5)),
            ('shift_plus', hp.uniform('shift_plus', -1., 0.)),

            ('n_fields', None),

            ('n_estimators', self.args.n_estimators),
            ('max_features', 1.0),
            ('max_samples', 0.2), # hp.uniform('max_samples', 0.1, 0.25)),
        ])
        with open(self.trial_dir / 'search_space.pkl', 'wb') as f:
            pickle.dump(ss, f)
        return ss

    def _maybe_resume(self):
        if not self.args.dir_for_resume:
            return
            
        resume_dir = self.base_dir / self.args.dir_for_resume
        trials_file = resume_dir / 'trials.pkl'
        search_space_file = resume_dir / 'search_space.pkl'

        if trials_file.exists():
            shutil.copy(trials_file, self.trial_dir / 'trials.pkl')
            logger.info(f"Copying and continuing trials from {trials_file}.")
        else:
            logger.warning(f"Trials file not found at {trials_file}!")
            return

        if search_space_file.exists():
            with open(search_space_file, 'rb') as f:
                self.search_space = pickle.load(f)
            shutil.copy(search_space_file, self.trial_dir / 'search_space.pkl') 
            logger.info(f"Using search space from {search_space_file}")
        else:
            logger.warning(f"""
                Search space does not exist in: {search_space_file}!\n
                Using search space from the executed script.
            """)

    def _save_params_and_wait(self, params):
        # Save parameters for each neuron
        for e in range(1, self.args.n_estimators + 1):
            for n in range(10):
                with open(self.data_dir / f'params_{e}_{n}.pkl', 'wb') as f:
                    pickle.dump(params, f)
        # Wait for output files with potentials (or spikes)
        expected_files = self._expected_files(params)
        missing = {f for f in expected_files if not (self.data_dir / f).exists()}
        # watch_for_files(self.data_dir, missing)
        while missing:
            time.sleep(30)
            missing = {f for f in expected_files if not (self.data_dir / f).exists()}
        return expected_files

    def _expected_files(self, params):
        files = []
        for n in range(10):
            for e in range(1, params['n_estimators']+1):
                files.append(f'neuron_{n}_ensemble_{e}_validation_outputs.npy')
        return files

    def _decode(self, params):
        """ Load outputs and calculate y_pred """
        outs = [np.load(self.data_dir / f) for f in self._expected_files(params)]
        spk = np.hstack(outs)
        np.save(self.data_dir / "full_output.npy", spk)
        # print(spk.shape)
        # if params['decoding_method']=='mahalanobis':
        #     _, _, y_train, _ = load_data("mnist")
        #     dec = MahalanobisDecoder(regularization=params.get('regularization',1e-6))
        #     dec.fit(spk, y_train[:len(spk)])
        #     return dec.predict(spk)
        if params['decoding_method']=='max':
            return spk.argmax(axis=1)
    
    def _save_best_weights(self):
        best_weights_dir = self.data_dir / "best_weights"
        best_weights_dir.mkdir(parents=True, exist_ok=True)
        for n in range(10):
            src = self.data_dir / f"neuron_{n}_weights.npy"
            dst = best_weights_dir / f"neuron_{n}_weights.npy"
            shutil.copy(src, dst)
    
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
            self._save_best_weights()
            self.best_score = score
        # Delete outputs related to the current optimization iteration
        clear_folder(self.data_dir, output_files)
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
    parser.add_argument("--job_id", default="unknown")
    parser.add_argument("--ho_bound", type=int, default=10000)
    parser.add_argument("--n_estimators", type=int, default=1)
    parser.add_argument("--dir_for_resume", default="")
    parser.add_argument("--encoding_method", default="frequency")
    parser.add_argument("--decoding_method", default="max")
    parser.add_argument("--validation_size", type=int, default=1000)
    args = parser.parse_args()

    exp = HPOExperiment(args)
    exp.optimize()

if __name__ == "__main__":
    main()
