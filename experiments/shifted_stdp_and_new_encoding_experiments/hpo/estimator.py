"""
Estimator in the case of classwise approach can be devided into `n_classes` parts,
because each of them will be trained on their own class data, which speeds up calculations.
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
from typing import List


def execute_script(
    script_path: Path,
    neuron: int,
    estimator: int,
    validation_size: int,
    job_id: str
):
    """
    Execute the continuous neuron scripts that wait for parameters in a specified directory 
    and execute training and validation using aforementioned parameters.
    Raises a CalledProcessError on non-zero exit codes.
    """

    try:
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(neuron),
                str(estimator),
                str(validation_size),
                job_id,
            ],
            check=True,
            capture_output=True,
            text=True
        )
        return 1.
    except subprocess.CalledProcessError as e:
        print(f"Error in script of neuron {neuron} of estimator {estimator}:")
        print(f"Error code: {e.returncode}")
        print(f"Error message: {e.stderr.strip()}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run neuron training scripts in parallel.")
    parser.add_argument("data_dir", type=Path, help="Path to the data directory, that is also used to determine `job ID`).")
    parser.add_argument("estimator_number", type=int, help="estimator number, which may specify training data.")
    parser.add_argument("validation_size", type=int, help="Last `validation_size` samples of train dataset are used for validation.")
    parser.add_argument("--script", type=Path, default=Path(__file__).parent / "neuron.py", help="Path to the neuron script.")
    parser.add_argument("--n_classes", type=int, default=10, help="Number of output neurons (classes).")
    args = parser.parse_args()

    job_id = args.data_dir.name
    errors = [] # list for errors in neuron scripts
    print("script: ", args.script)

    with ThreadPoolExecutor(max_workers=args.n_classes) as executor:
        futures = {
            executor.submit(
                execute_script,
                args.script,
                i,
                args.estimator_number,
                args.validation_size,
                job_id
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
