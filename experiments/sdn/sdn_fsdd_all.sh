#! /bin/sh

M=30 E=1 P="stdp_nn_symm_synapse" sh $PWD/experiments/sdn/run_sdn_fsdd.sh
M=30 E=1 P="stdp_tanh_synapse" sh $PWD/experiments/sdn/run_sdn_fsdd.sh
M=30 E=1 P="stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse" sh $PWD/experiments/sdn/run_sdn_fsdd.sh

M=30 E=1 P="stdp_nn_symm_synapse" sh $PWD/experiments/sdn/run_sdn_fsdd_base.sh
M=30 E=1 P="stdp_tanh_synapse" sh $PWD/experiments/sdn/run_sdn_fsdd_base.sh
M=30 E=1 P="stdp_gaussian_times_linear_with_separate_exp_r_dependence_synapse" sh $PWD/experiments/sdn/run_sdn_fsdd_base.sh
