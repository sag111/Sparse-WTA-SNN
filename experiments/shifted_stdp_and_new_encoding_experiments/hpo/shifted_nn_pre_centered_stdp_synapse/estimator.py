"""
Estimator is a full network model.
In case of classwise approach Estimator can be devided into `n_classes` parts,
because each of them will be trained on their own class data, which will speed up calculations.
"""

import sys
path_to_remove = '/s/ls4/users/romanrybka/YD/fsnn-classifiers'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)

import argparse
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
# import logging
from typing import List


def split_list(lst, n_parts):
    k, m = divmod(len(lst), n_parts)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_parts)]


def execute_script(
    script_path: Path,
    neuron: int,
    ensemble: int,
    validation_size: int,
    job_id: str,
    # cpu_subset: List[int],
):
    """
    Execute the continuous neuron scripts that wait for parameters in a specified directory 
    and execute training and validation using aforementioned parameters.
    Raises a CalledProcessError on non-zero exit codes.
    """

    try:
        subprocess.run(
            [
                # "taskset", "-c", ",".join(map(str, cpu_subset)),
                sys.executable,
                str(script_path),
                str(neuron),
                str(ensemble),
                str(validation_size),
                job_id,
                # str(len(cpu_subset)), # n_jobs
            ],
            check=True,
            capture_output=True,
            text=True
        )
        return 1.
    except subprocess.CalledProcessError as e:
        print(f"Error in script of neuron {neuron} of ensemble {ensemble}:")
        print(f"Error code: {e.returncode}")
        print(f"Error message: {e.stderr.strip()}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run neuron training scripts in parallel.")
    parser.add_argument("data_dir", type=Path, help="Path to the data directory, that is also used to determine `job ID`).")
    parser.add_argument("ensemble_number", type=int, help="Ensemble number, which may specify training data.")
    parser.add_argument("validation_size", type=int, help="Last `validation_size` samples of train dataset are used for validation.")
    parser.add_argument("--script", type=Path, default=Path(__file__).parent / "neuron.py", help="Path to the neuron script.")
    parser.add_argument("--n_classes", type=int, default=10, help="Number of output neurons (classes).")
    args = parser.parse_args()

    job_id = args.data_dir.name

    # # Logging config
    # job_task_id = os.environ.get('SLURM_JOB_ID') + '_' + os.environ.get('SLURM_ARRAY_TASK_ID')
    # log_path = Path(f'/s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/shifted_nn_symm_stdp_synapse/logs/{job_task_id}/{job_task_id}.log')
    # log_path.parent.mkdir(parents=True, exist_ok=True)
    # logging.basicConfig(
    #     filename=log_path,
    #     level=logging.INFO,
    #     format='%(asctime)s - %(levelname)s - %(message)s'
    # )   

    # allowed = sorted(os.sched_getaffinity(0))
    # if len(allowed) < args.n_classes:
    #     raise RuntimeError(f"Not enough cores: need {args.n_classes}, but available {len(allowed)}")

    # cores_per_neuron = split_list(allowed, args.n_classes)
    # logging.info(f"cores_per_neuron: {cores_per_neuron}")
 
    errors = [] # list for errors in neuron scripts

    with ThreadPoolExecutor(max_workers=args.n_classes) as executor:
        futures = {
            executor.submit(
                execute_script,
                args.script,
                i,
                args.ensemble_number,
                args.validation_size,
                job_id,
                # cores_per_neuron[i],
            ): i for i in range(args.n_classes)
        }

        for future in as_completed(futures):
            try:
                result = future.result()
            except subprocess.CalledProcessError:
                errors.append(futures[future])

    if errors:
        print(f"Some neuron scripts raised errors:\n\n {errors}")
        raise RuntimeError("Some neuron scripts raised errors.")


if __name__ == "__main__":
    main()
