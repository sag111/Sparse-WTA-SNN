#! /bin/sh

#SBATCH -D /s/ls4/users/romanrybka/YD/fsnn-classifiers/
#SBATCH -o /s/ls4/users/romanrybka/YD/fsnn-classifiers/logs/ccn_f3_%j.out
#SBATCH -e /s/ls4/users/romanrybka/YD/fsnn-classifiers/logs/ccn_f3_%j.err
#SBATCH -p hpc4-el7-3d
#SBATCH --cpus-per-task 56
#SBATCH -n 1
#SBATCH -t 72:00:00

source /s/ls4/users/romanrybka/anaconda3/bin/activate
module load openmpi
conda activate /s/ls4/groups/g0126/conda_envs/nest

echo env activated
echo start script

export PYTHONPATH=/s/ls4/users/romanrybka/YD/fsnn-classifiers/:$PYTHONPATH

export MPLCONFIGDIR=/s/ls4/users/romanrybka/.config/matplotlib

echo PYTHONPATH

python $PWD/experiments/ccn/ccn_mnist_f3.py --epochs 1 --plasticity "stdp_nn_restr_synapse" --dataset "mnist" --n_estimators 51 --drop_extra True --n_mfcc 40

echo end script
