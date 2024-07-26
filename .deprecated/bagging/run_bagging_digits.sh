#! /bin/sh

conda activate sparse-snn-wta

echo env activated
echo start script

python $PWD/experiments/bagging/bagging_digits.py --epochs $E --plasticity $P

echo end script
