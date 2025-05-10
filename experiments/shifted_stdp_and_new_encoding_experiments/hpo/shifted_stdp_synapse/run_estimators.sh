#!/bin/sh
#SBATCH -D /s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/shifted_stdp_synapse
#SBATCH -o /s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/shifted_stdp_synapse/logs/%A_%a.out
#SBATCH -e /s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/shifted_stdp_synapse/logs/%A_%a.err
#SBATCH -p hpc4-el7-3d
#SBATCH -n 1
#SBATCH --array=1
#SBATCH --cpus-per-task 10

validation_size=1000
ensemble_number=$SLURM_ARRAY_TASK_ID

# Firstly, execute `sbatch run_hpo_ensemble.sh' that generates `data_dir.txt`
until [ -f data_dir.txt ]; do
  sleep 1
done
data_dir=$(<data_dir.txt)

# Delete file in the last job in array
if [ "$ensemble_number" -eq "$SLURM_ARRAY_TASK_MAX" ]; then
  rm -f data_dir.txt
fi

source /s/ls4/users/selibrin/anaconda3/bin/activate
conda activate /s/ls4/groups/g0126/conda_envs/nest

(
    . /s/ls4/groups/g0126/opt/nest3.6/bin/nest_vars.sh
    python /s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/estimator.py \
      "$data_dir" \
      "$ensemble_number" \
      "$validation_size" \
      --script="/s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/shifted_stdp_synapse/neuron.py"
)
