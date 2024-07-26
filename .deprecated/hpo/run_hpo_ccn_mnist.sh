#! /bin/sh

#SBATCH -D /s/ls4/users/romanrybka/YD/fsnn-classifiers/
#SBATCH -o /s/ls4/users/romanrybka/YD/fsnn-classifiers/logs/ccn_mnist_hpo_%j.out
#SBATCH -e /s/ls4/users/romanrybka/YD/fsnn-classifiers/logs/ccn_mnist_hpo_%j.err
#SBATCH -p hpc4-el7-3d
#SBATCH --cpus-per-task 56
#SBATCH -n 1
#SBATCH -t 72:00:00

# P should be specified as environment variable

source /s/ls4/users/romanrybka/anaconda3/bin/activate
module load openmpi
conda activate /s/ls4/groups/g0126/conda_envs/nest

echo env activated
echo start script

echo "Dataset: MNIST1000"
echo "Plasticity: $P"

export PYTHONPATH=/s/ls4/users/romanrybka/YD/fsnn-classifiers/:$PYTHONPATH

export MPLCONFIGDIR=/s/ls4/users/romanrybka/.config/matplotlib

echo $PYTHONPATH

if [ $RESUME ]; then
  python $PWD/experiments/hpo/ccn_mnist_hpo.py --plasticity $P --dataset $D --ho_bound 600 --resume
else
  python $PWD/experiments/hpo/ccn_mnist_hpo.py --plasticity $P --dataset $D --ho_bound 600
fi

echo end script
