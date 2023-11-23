#! /bin/sh

E=1 P="stdp_nn_symm_synapse" sh $PWD/experiments/bagging/run_bagging_digits.sh
E=1 P="stdp_tanh_synapse" sh $PWD/experiments/bagging/run_bagging_digits.sh
E=1 P="stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse" sh $PWD/experiments/bagging/run_bagging_digits.sh
