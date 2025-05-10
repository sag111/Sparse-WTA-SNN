#!/bin/sh
#SBATCH -D /s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/single_spk_input_multi_spk_output
#SBATCH -o /s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/single_spk_input_multi_spk_output/logs/%j.out
#SBATCH -e /s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/single_spk_input_multi_spk_output/logs/%j.err
#SBATCH -p hpc4-el7-3d
#SBATCH -n 1
#SBATCH --cpus-per-task 1

DIR_FOR_RESUME="${1:-}"

source /s/ls4/users/selibrin/anaconda3/bin/activate
conda activate /s/ls4/groups/g0126/conda_envs/nest

(
    . /s/ls4/groups/g0126/opt/nest3.6/bin/nest_vars.sh
    python /s/ls4/users/selibrin/spiking_researches/correlation_encoding_and_training/mnist/hpo/single_spk_input_multi_spk_output/hpo_experiment.py \
    --job_id=$SLURM_JOB_ID \
    --n_estimators=1 \
    --dir_for_resume=$DIR_FOR_RESUME
)