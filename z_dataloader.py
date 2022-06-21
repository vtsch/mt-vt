import numpy as np
import pandas as pd
import random

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


def load_data(file_name_train, file_name_test):
    """
    Loads data from a csv file and returns np.array of data
    """
    train_raw = np.loadtxt(file_name_train, delimiter=",",dtype=float)
    test_raw = np.loadtxt(file_name_test, delimiter=",",dtype=float)

    #shuffle rows
    np.random.shuffle(train_raw)
    np.random.shuffle(test_raw)

    print("raw data loaded")

    return train_raw, test_raw

def upsample(data, n_samples, n_classes):
    #upsample to number of samples of the 2 classes with most samples (0 & 4)
    df_0=data[data[:, 187]==0]
    df_1=data[data[:, 187]==1]
    df_2=data[data[:, 187]==2]
    df_3=data[data[:, 187]==3]
    df_4=data[data[:, 187]==4]

    df_0_upsample=resample(df_0,replace=True,n_samples=n_samples)
    df_1_upsample=resample(df_1,replace=True,n_samples=n_samples)
    df_2_upsample=resample(df_2,replace=True,n_samples=n_samples)
    df_3_upsample=resample(df_3,replace=True,n_samples=n_samples)
    df_4_upsample=resample(df_4,replace=True,n_samples=n_samples)
    
    data_upsampled_twoclasses = np.concatenate([df_0_upsample, df_4_upsample])
    np.random.shuffle(data_upsampled_twoclasses)
    data_upsampled_fiveclasses = np.concatenate([df_0_upsample, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])
    np.random.shuffle(data_upsampled_fiveclasses)

    if n_classes == 2:
        return data_upsampled_twoclasses
    elif n_classes == 5:
        return data_upsampled_fiveclasses

def loadfiveclusters(train_raw, test_raw):

    train_raw, test_raw = load_data(train_raw, test_raw) 
    #reduce nr. of samples
    train = upsample(train_raw, 2000, 5)
    test = upsample(test_raw, 1000, 5)

    #split to X and y
    X_train = train[:, 0:-1]
    y_train = train[:, -1]

    X_test = test[:, 0:-1]
    y_test = test[:, -1]

    return X_train, X_test, y_train, y_test


def loadtwoclusters(train_raw, test_raw):

    train_raw, test_raw = load_data(train_raw, test_raw)

    train = upsample(train_raw, 2000, 2)
    test = upsample(test_raw, 1000, 2)

    X_train = train[:, 0:-1]
    y_train = train[:, -1]

    X_test = test[:, 0:-1]
    y_test = test[:, -1]

    return X_train, X_test, y_train, y_test

def create_irregular_ts(data):
    
    # random boolean mask for which values will be changed
    replace_rate = 0.3
    mask = np.random.choice([0, 1], size=data.shape, p=((1-replace_rate),replace_rate)).astype(np.bool)

    # random matrix the same shape
    r = np.zeros(shape=data.shape)

    # use  mask to replace values in input array
    data[mask] = r[mask]

    return data

def create_pandas_df(train, test):
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    train = train.values
    test = test.values

    return train, test


def reshape_lstm(X_train, X_test, timesteps, n_features):
    #reshape to fit in lstm - need n_samples x timesteps/lookback x n_features

    train_lstm = X_train.reshape(-1, timesteps, n_features)
    test_lstm = X_test.reshape(-1, timesteps, n_features)
    return train_lstm, test_lstm

"""
def create_torch_df(train, test):
    # creating tensor from pd df
    train_tensor = torch.tensor(train)
    test_tensor = torch.tensor(test)

    train_dataset = TensorDataset(train_tensor) 
    test_dataset = TensorDataset(test_tensor) 

    train_loader = DataLoader(train_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # printing out result
    print(train_tensor.shape)
    return train_dataset, test_dataset, train_loader, test_loader
"""

class ECGDataset(Dataset):

    def __init__(self, df):
        self.df = df
        self.data_columns = self.df.columns[:-2].tolist()

    def __getitem__(self, idx):
        signal = self.df.loc[idx, self.data_columns].astype('float32')
        signal = torch.FloatTensor(np.array([signal.values]))         
        target = torch.LongTensor(np.array(self.df.loc[idx, 'class']))
        return signal, target

    def __len__(self):
        return len(self.df)

def generate_train_test(data):
    train_df, val_df = train_test_split(
        data, test_size=0.15, random_state=42, stratify=data['class']
    )
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    return train_df, val_df

def get_dataloader(train_data, phase: str, batch_size: int) -> DataLoader:
    '''
    Dataset and DataLoader.
    Parameters:
        phase: training or validation phase.
        batch_size: data per iteration.
    Returns:
        data generator
    '''
    train_df, val_df = generate_train_test(train_data)
    df = train_df if phase == 'train' else val_df
    dataset = ECGDataset(df)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
    return dataloader