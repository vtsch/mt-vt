import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


def create_irregular_ts(data):
    # random boolean mask for which values will be changed
    replace_rate = 0.3
    mask = np.random.choice([0, 1], size=data.shape, p=((1-replace_rate),replace_rate)).astype(np.bool)

    # random matrix the same shape
    r = np.zeros(shape=data.shape)

    # use  mask to replace values in input array
    data[mask] = r[mask]

    return data

def load_raw_data_to_pd(file_name_train, file_name_test):
    df_mitbih_train = pd.read_csv(file_name_train, header=None)
    df_mitbih_test = pd.read_csv(file_name_test, header=None)
    #df_mitbih = pd.concat([df_mitbih_train, df_mitbih_test], axis=0)
    df_mitbih_train.rename(columns={187: 'class'}, inplace=True)
    df_mitbih_test.rename(columns={187: 'class'}, inplace=True)
    #print(df_mitbih.head(5))

    return df_mitbih_train, df_mitbih_test

def upsample_data(df_mitbih, n_clusters, sample_size):
    #select sample_size samples from each class
    df = pd.DataFrame()
    for i in range(n_clusters):
        df = pd.concat([df, df_mitbih.loc[df_mitbih['class'] == i].sample(n=sample_size)])
    #shuffle rows of df and remove index column
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)  
    #print(df.head(5))

    return df


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

def get_test_dataloader(test_data, batch_size: int) -> DataLoader:
    '''
    Dataset and DataLoader.
    Parameters:
        batch_size: data per iteration.
    Returns:
        data generator
    '''
    dataset = ECGDataset(test_data)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
    return dataloader