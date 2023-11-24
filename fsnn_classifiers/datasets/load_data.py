from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
import numpy as np

import csv
import librosa
import pandas as pd

import os

def extract_wav_features(soundFilesFolder, csvFileName, n_mfcc = 20):
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

def preprocess_fsdd(csvFileName, drop_extra=True):
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

def load_data(dataset, 
              test_size=0.25, 
              n_mfcc=30, 
              drop_extra=True,
              dhl_syn="stdp_nn_symm_synapse",
              seed=42):
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

    else:
        raise NotImplementedError("Wrong dataset name.")
    
    return X_train, X_test, y_train, y_test
   