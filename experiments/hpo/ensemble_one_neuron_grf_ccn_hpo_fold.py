import os
os.environ["PYNEST_QUIET"] = "1"

import sys

# import nest
# try:
#     nest.Install("diehl_neuron_module")
# except Exception as e:
#     print(e)

path_to_remove = '/s/ls4/users/romanrybka/YD/fsnn-classifiers'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)
    
sys.path.insert(0, '/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN')

from fsnn_classifiers.components.networks.correlation_classwise_network import CorrelationClasswiseNetwork as BaseNetwork
from fsnn_classifiers.components.networks.correlation_classwise_anti_network import CorrelationClasswiseNetwork as AntiNetwork
from fsnn_classifiers.components.networks.correlation_classwise_network_one_spike import CorrelationClasswiseNetwork as OneSpikeNetwork
from fsnn_classifiers.components.decoding.own_rate_population_decoder import OwnRatePopulationDecoder
from fsnn_classifiers.components.preprocessing.grf import GRF
from fsnn_classifiers.components.preprocessing.pooling import Pooling
from fsnn_classifiers.optimization.hpo import adjust
from fsnn_classifiers.datasets.load_data import load_data
from fsnn_classifiers.components.preprocessing.center_and_normalize import center_and_normalize_vec
from fsnn_classifiers.components.preprocessing.dog import DoG

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.utils import shuffle

import numpy as np

from typing import OrderedDict

from hyperopt import hp

import argparse
import pickle
import time


def swap_zeros(arr, digit):
    return np.array([digit if x == 0 else (0 if x == digit else x) for x in arr])


def get_block_indices(X, ensemble_number, block_height=7, block_width=7, image_size=28):
    # чтобы нумерация начиналась с 0
    ensemble_number = ensemble_number - 1

    # Проверка размера изображения
    assert X.shape[1] == image_size * image_size, f"Размер изображения должен быть {image_size}x{image_size}"

    # Количество блоков по горизонтали и вертикали
    blocks_per_row = int(np.ceil(image_size / block_width))
    blocks_per_col = int(np.ceil(image_size / block_height))
    
    # Индексы для изображения
    feat_idx = np.arange(0, image_size * image_size).reshape(image_size, image_size)
    
    # Индекс строки и столбца для блока
    row = ensemble_number // blocks_per_row  # Строка блока
    col = ensemble_number % blocks_per_row  # Столбец блока
    
    # Получаем индексы текущего блока
    block_indices = feat_idx[
        row * block_height: (row + 1) * block_height,
        col * block_width: (col + 1) * block_width
    ].ravel()
    
    return block_indices


if __name__ == "__main__":

    neuron = sys.argv[1]
    ensemble_number = sys.argv[2]
    validation_size = sys.argv[3]
    job_id = sys.argv[4]

    path = f'/s/ls4/users/selibrin/spiking_researches/Sparse-WTA-SNN/experiments/hpo/data_for_one_neuron/{job_id}'

    with open(f'{path}/params.pkl', 'rb') as f:   
        params = pickle.load(f)

    # Выбор сети в зависимости от наличия ключа 'anti_lr' в словаре
    CorrelationClasswiseNetwork = AntiNetwork if 'anti_lr' in params \
    else (OneSpikeNetwork if params['encoding_method'] == 'time' else BaseNetwork)

    params["class_name"] = int(neuron)

    X_train, X_test, y_train, y_test = load_data("mnist")
    # создаем валидационное множество
    X_validation, y_validation = X_train[-int(validation_size):], y_train[-int(validation_size):]
    
    X_train_untouched, y_train_untouched = X_train, y_train
    if params["run_on_test_mnist"]:
        X_train_for_run = X_train_untouched

    if "anti_lr" not in params:
        # выбираем примеры своего класса
        X_train, y_train = X_train[y_train == int(neuron)], y_train[y_train == int(neuron)]
        # так как создается один нейрон в сетке, метка его класса должна быть нулем из-за реализации
        y_train = np.zeros_like(y_train)
    else:
        X_train = [X_train[y_train == cl][:int(params["max_samples"] * X_train_untouched.shape[0] / 10)] for cl in range(10)]
        y_train = [y_train[y_train == cl][:int(params["max_samples"] * X_train_untouched.shape[0] / 10)] for cl in range(10)]
        # Чередуем примеры: сначала класс 0, потом класс 1, и так до 9
        X_train = np.vstack([np.array(batch) for batch in zip(*X_train)])
        y_train = np.hstack([np.array(batch) for batch in zip(*y_train)])

    if not params.get("clustering"):
        # выбираем случайное подмножество примеров на которых нейрон будет обучаться
        random_state = int(ensemble_number)
        X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
        max_samples = int(params["max_samples"] * len(y_train_untouched))
    else:
        clusters_of_the_class = np.load(f'{path}/clusters_of_class_{neuron}.npy')
        X_train = X_train[clusters_of_the_class == int(ensemble_number) - 1]
        y_train = y_train[clusters_of_the_class == int(ensemble_number) - 1]

    # выбираем случайное подножество признаков на которых нейрон будет обучаться и валидироваться
    if params["max_features"] < 1.:
        features = list(range(X_train.shape[1]))
        np.random.seed(random_state)
        np.random.shuffle(features)
        max_features = params["max_features"]
        selected_features = features[:int(max_features * len(features))]
        X_train = X_train[:, selected_features]
        X_validation = X_validation[:, selected_features]
        if params["run_on_test_mnist"]:
            X_train_for_run = X_train_for_run[:, selected_features]
        X_test = X_test[:, selected_features]

    if 'reversed_image' in params and params['reversed_image']:
        X_train = X_train.max() - X_train
        X_validation = X_validation.max() - X_validation
    
    # предобработка полями
    if "n_fields" in params and params["n_fields"] is not None:
        prep = GRF(params["n_fields"])
        X_train = prep.fit_transform(X_train)
        X_validation = prep.transform(X_validation)
        X_test = prep.transform(X_test)

        if params["run_on_test_mnist"]:
            X_train_for_run = prep.transform(X_train_for_run)

    # нормировка + сетка
    # nrm = Normalizer(norm=params["norm"]) if params["norm"] != "std" else StandardScaler()

    ''' 
        Удаляем параметры n_estimators, а также max_features, max_samples из params, 
        поскольку мы их уже использовали для предобработки датасета 
    '''
    del params["n_estimators"]
    del params["max_features"]
    del params["max_samples"]

    ccn = CorrelationClasswiseNetwork(
        n_estimators=1,
        max_features=1.,
        max_samples=1.,
        # epochs=1,
        corr_time=0.0,
        # t_ref=0.0,
        # ref_seq_interval=2,
        quiet=True,
        sample_norm=1,
        # resolution=0.5,
        w_inh=None,
        # w_init=0.0,
        weight_normalization=None,
        early_stopping=False,
        n_jobs=1,
        job_id=job_id,
        ensemble_number=ensemble_number,
        **params,
    )

    np.save(path + f'/validation_samples.npy', X_validation)
    np.save(path + f'/train_samples.npy', X_train)

    # обучение + валидация
    ccn.fit(X_train, y_train)

    if params['run_on_test_mnist']:
        train_outputs = ccn.transform(X_train_for_run[:10000]) \
        if params["decoding_method"] == 'mahalanobis' else None
        test_outputs = ccn.transform(X_test)

        if train_outputs:
            np.save(path + f'/neuron_{neuron}_from_ensemble_{ensemble_number}_train_outputs.npy', train_outputs)
        np.save(path + f'/neuron_{neuron}_from_ensemble_{ensemble_number}_test_outputs.npy', test_outputs)
    else:
        validation_outputs = ccn.transform(X_validation)
        np.save(path + f'/neuron_{neuron}_from_ensemble_{ensemble_number}_validation_outputs.npy', validation_outputs)

    # saving weights and selected features
    os.makedirs(path + f'/weights_and_features/ensemble_{ensemble_number}/neuron_{neuron}/', exist_ok=True)
    np.save(path + f'/weights_and_features/ensemble_{ensemble_number}/neuron_{neuron}/weights.npy', ccn.weights_)

    print(1)