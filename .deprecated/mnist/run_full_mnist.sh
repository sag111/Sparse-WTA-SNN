#! /bin/sh

#SBATCH -D /s/ls4/users/romanrybka/YD/fsnn-classifiers/
#SBATCH -o /s/ls4/users/romanrybka/YD/fsnn-classifiers/logs/ccn_mnist_hpo_%j.out
#SBATCH -e /s/ls4/users/romanrybka/YD/fsnn-classifiers/logs/ccn_mnist_hpo_%j.err
#SBATCH -p hpc4-el7-3d
#SBATCH --cpus-per-task 56
#SBATCH -n 1
#SBATCH -t 72:00:00

source /s/ls4/users/romanrybka/anaconda3/bin/activate
module load openmpi
conda activate /s/ls4/groups/g0126/conda_envs/nest

echo env activated
echo start script

echo "Dataset: MNIST1000"
echo "Plasticity: $P"
echo "train time: $TRAIN_TIME, test time: $TEST_TIME, estimators: $N, max samples: $MAX_SAMPLES"

export PYTHONPATH=/s/ls4/users/romanrybka/YD/fsnn-classifiers/:$PYTHONPATH

export MPLCONFIGDIR=/s/ls4/users/romanrybka/.config/matplotlib

echo $PYTHONPATH


python $PWD/experiments/mnist/mnist_full_ccn.py $P --decoding "frequency" --time $TRAIN_TIME --test_time $TEST_TIME --epochs 1 --full_ds --max_train $MAX_TRAIN --max_features 0.7 --n_estimators $N --max_samples $MAX_SAMPLES