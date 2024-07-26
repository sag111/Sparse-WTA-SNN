#! /bin/sh


D="fsdd" E=1 N=11 P="stdp_nn_restr_synapse" sbatch $PWD/experiments/ccn/run_ccn_mnist.sh
D="fsdd" E=1 N=11 P="stdp_nn_restr_synapse" sbatch $PWD/experiments/ccn/run_ccn_mnist_filter.sh
#D="iris" E=1 N=1 P="stdp_tanh_synapse" sbatch $PWD/experiments/ccn/run_ccn_sklearn.sh
#D="iris" E=1 N=1 P="stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse" sbatch $PWD/experiments/ccn/run_ccn_sklearn.sh