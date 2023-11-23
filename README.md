# Sparse_SNN WTA 

1. create evironment
```
    conda create --name sparse-snn-wta python=3.11 cmake cython libtool numpy scikit-learn pandas tqdm
    conda activate sparse-snn-wta
    conda install -c conda-forge librosa
```
2. download and install nest https://github.com/nest/nest-simulator/releases/tag/v3.5
3. install memrisitve plasticity
```    
    cd nest_modules/tanh_stdp_nest_module
    sh install.sh
    source */nest_vars.sh (like in hint)
```
4. install neuron model
```
    pip install nestml
    cd nest_modules/diehl_neuron_module
    sh install.sh
    source */nest_vars.sh (like in hint)
```
5. install fsnn_classifiers
```
    python setup.py install
```
6. run scripts from experiments
7. ...
8. PROFIT
