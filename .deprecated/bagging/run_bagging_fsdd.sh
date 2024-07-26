#! /bin/sh

conda activate sparse-snn-wta

echo env activated
echo start script

python $PWD/experiments/bagging/bagging_fsdd.py --epochs $E --plasticity $P --n_mfcc $M

echo end script
