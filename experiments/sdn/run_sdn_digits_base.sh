#! /bin/sh

conda activate sparse-snn-wta

echo env activated
echo start script

python $PWD/experiments/sdn/sdn_digits_base.py --epochs $E --plasticity $P

echo end script
