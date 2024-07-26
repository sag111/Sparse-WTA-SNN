#! /bin/sh

P="stdp_nn_restr_synapse" MAX_TRAIN=10000 MAX_SAMPLES=0.1 TRAIN_TIME=1000 TEST_TIME=2000 N=21 sbatch experiments/mnist/run_full_mnist.sh
P="stdp_nn_restr_synapse" MAX_TRAIN=10000 MAX_SAMPLES=0.1 TRAIN_TIME=1000 TEST_TIME=2000 N=41 sbatch experiments/mnist/run_full_mnist.sh
P="stdp_nn_restr_synapse" MAX_TRAIN=20000 MAX_SAMPLES=0.1 TRAIN_TIME=1000 TEST_TIME=2000 N=21 sbatch experiments/mnist/run_full_mnist.sh
P="stdp_nn_restr_synapse" MAX_TRAIN=20000 MAX_SAMPLES=0.05 TRAIN_TIME=1000 TEST_TIME=2000 N=21 sbatch experiments/mnist/run_full_mnist.sh
P="stdp_nn_restr_synapse" MAX_TRAIN=10000 MAX_SAMPLES=0.1 TRAIN_TIME=1500 TEST_TIME=3000 N=21 sbatch experiments/mnist/run_full_mnist.sh