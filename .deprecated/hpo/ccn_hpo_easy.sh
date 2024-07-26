#! /bin/sh

datasets=('iris' 'cancer' 'fsdd')
rules=('stdp_nn_restr_synapse')

if [ $RESUME ]; then
  for d in "${datasets[@]}"
  do
    for p in "${rules[@]}"
    do
      D=$d P=$p RESUME=true sbatch run_hpo_ccn_easy.sh
    done
  done
else
  for d in "${datasets[@]}"
  do
    for p in "${rules[@]}"
    do
      D=$d P=$p sbatch run_hpo_ccn_easy.sh
    done
  done
fi


