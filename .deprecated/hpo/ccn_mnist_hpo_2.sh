#! /bin/sh

datasets=('mnist1000', 'fsdd')
rules=('stdp_nn_restr_synapse')

if [ $RESUME ]; then
  for d in "${datasets[@]}"
  do
    for p in "${rules[@]}"
    do
      P=$p D=$d RESUME=true sbatch run_hpo_ccn_mnist_2.sh
    done
  done
else
  for d in "${datasets[@]}"
  do
    for p in "${rules[@]}"
    do
      P=$p D=$d sbatch run_hpo_ccn_mnist_2.sh
    done
  done
fi


