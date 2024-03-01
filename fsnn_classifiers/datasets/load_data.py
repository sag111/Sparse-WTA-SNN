import numpy as np

import os
import pickle
import csv
import librosa
import pandas as pd

import struct
from array import array
from collections import defaultdict

from sklearn.datasets import (load_iris, 
                              load_breast_cancer, 
                              load_digits, 
                              load_diabetes, 
                              load_wine)
from sklearn.model_selection import train_test_split

class MnistDataloader(object):
    def __init__(self, 
                 training_images_filepath: str,
                 training_labels_filepath: str,
                 test_images_filepath: str, 
                 test_labels_filepath: str):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath: str, labels_filepath: str) -> tuple:        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

def extract_wav_features(soundFilesFolder: str, csvFileName: str, n_mfcc: int = 20):
    print("The features of the files in the folder "+soundFilesFolder+" will be saved to "+csvFileName)
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, n_mfcc+1):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    print('CSV Header: ', header)
    file = open(csvFileName, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(header)
    genres = '1 2 3 4 5 6 7 8 9 0'.split()
    for filename in os.listdir(soundFilesFolder):
        number = f'{soundFilesFolder}/{filename}'
        y, sr = librosa.load(number, mono=True, duration=30)
        # remove leading and trailing silence
        y, index = librosa.effects.trim(y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = n_mfcc)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        writer.writerow(to_append.split())
    file.close()
    print("End of extractWavFeatures")

def preprocess_fsdd(csvFileName: str, drop_extra: bool = True) -> pd.DataFrame:
    print(csvFileName+ " will be preprocessed")
    data = pd.read_csv(csvFileName)
    data['number'] = data['filename'].str[:1]
    #Dropping unnecessary columns
    data = data.drop(['filename'],axis=1)
    data = data.drop(['label'],axis=1)
    if drop_extra:
        data = data.drop(['chroma_stft'],axis=1)
        data = data.drop(['rmse'],axis=1)
        data = data.drop(['spectral_centroid'],axis=1)
        data = data.drop(['spectral_bandwidth'],axis=1)
        data = data.drop(['rolloff'],axis=1)
        data = data.drop(['zero_crossing_rate'],axis=1)

    print("Preprocessing is finished")
    return data

def load_data(dataset: str, 
              test_size:float = 0.25, 
              n_mfcc: int = 30, 
              drop_extra: bool = True,
              max_train: int = 60000,
              seed: int = 42) -> tuple:
    if dataset == "iris":
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)
    elif dataset == "cancer":
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)
    elif dataset == "digits":
         X, y = load_digits(return_X_y=True)
         X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)
    elif dataset == "diabetes":
         X, y = load_diabetes(return_X_y=True)
         X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)
    elif dataset == "wine":
         X, y = load_wine(return_X_y=True)
         X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)
    elif dataset == "fsdd":
        cur_path = os.path.dirname(__file__)
        
        path_to_full_fsdd=os.path.join(cur_path, "../../fsdd_data/recordings")
        csv_root=os.path.join(cur_path,"_fsdd_csv/")
        if not os.path.exists(f"{csv_root}/full_train.csv"):
            assert os.path.exists(f"{path_to_full_fsdd}/train/0_george_5.wav"), "Place the FSDD audio files in the root directory!"
            extract_wav_features(f"{path_to_full_fsdd}/train/", f"{csv_root}/full_train.csv", n_mfcc)
            extract_wav_features(f"{path_to_full_fsdd}/test/", f"{csv_root}/full_test.csv", n_mfcc)

        train_data = preprocess_fsdd(f"{csv_root}/full_train.csv", drop_extra)
        test_data = preprocess_fsdd(f"{csv_root}/full_test.csv", drop_extra)

        X_train = np.asarray(train_data.iloc[:, :-1]).astype(np.float32)
        y_train = np.asarray(train_data.iloc[:, -1]).astype(np.int32)

        X_test = np.asarray(test_data.iloc[:, :-1]).astype(np.float32)
        y_test = np.asarray(test_data.iloc[:, -1]).astype(np.int32)

    elif dataset == "mnist":
        loader = MnistDataloader(
            training_images_filepath = f"{os.path.dirname(__file__)}/_mnist_data/full/train-images.idx3-ubyte",
            training_labels_filepath = f"{os.path.dirname(__file__)}/_mnist_data/full/train-labels.idx1-ubyte",
            test_images_filepath = f"{os.path.dirname(__file__)}/_mnist_data/full/t10k-images.idx3-ubyte",
            test_labels_filepath = f"{os.path.dirname(__file__)}/_mnist_data/full/t10k-labels.idx1-ubyte",
        )

        (X_train, y_train), (X_test, y_test) = loader.load_data()
        X_train = np.array(X_train).reshape((-1, 784))
        X_test = np.array(X_test).reshape((-1, 784))
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        samples = defaultdict(list)
        labels = defaultdict(list)

        for x, y in zip(X_train, y_train):
            if len(samples[y]) < int(max_train/10):
                samples[y].append(x)
                labels[y].append(y)
            
        X_train = []
        y_train = []
        for sample, label in zip(samples.values(), labels.values()):
            X_train.extend(sample)
            y_train.extend(label)
        X_train = np.array(X_train)
        y_train = np.array(y_train)

    elif dataset == "mnist1000":
        mnist_path = f"{os.path.dirname(__file__)}/_mnist_data/mini-mnist-1000.pickle"

        with open(mnist_path, 'rb') as fp:
            data = pickle.load(fp)

        X_train, y_train = np.array(data['images']).reshape(-1, 28*28), np.array(data['labels'])
        X_test, y_test = np.array(data['images']).reshape(-1, 28*28), np.array(data['labels'])

    else:
        raise NotImplementedError("Wrong dataset name.")
    
    return X_train, X_test, y_train, y_test
   