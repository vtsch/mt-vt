import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset, DataLoader
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from preprocess import generate_split, normalize


# ---- Dataloader ----

class PSADataset(Dataset):
    def __init__(self, config, data):
        self.df = data
        self.data_columns_psa = self.df.columns[0:6].tolist()
        if config.DELTATIMES:
            self.data_columns_ts = self.df.columns[7:13].tolist()
        else:
            self.data_columns_ts = self.df.columns[6:12].tolist()

    def __getitem__(self, idx):
        target = torch.tensor(np.array(self.df.loc[idx, 'pros_cancer']))   
        signal = self.df.loc[idx, self.data_columns_psa].astype('float32')
        signal = torch.tensor(np.array([signal.values]))  
        index = self.df.loc[idx, self.data_columns_ts]
        index = torch.tensor(np.array([index.values.astype("int64")]))
        #print("target", target, "index", index, "signal", signal)
        return signal, target, index

    def __len__(self):
        return len(self.df)

def get_dataloader(config, data, phase: str) -> DataLoader:
    '''
    Parameters:
        config: config file
        data: dataframe
        phase: training, validation or test
    Returns:
        data generator
    '''
    train_df, test_df = generate_split(data)
    train_df, val_df = generate_split(train_df)

    if phase == 'train':
        df = train_df
    elif phase == 'val':
        df = val_df
    elif phase == 'test':
        df = test_df
    else:
        raise ValueError('phase must be either train, val or test')

    dataset = PSADataset(config, df)
    
    print(f'{phase} data shape: {dataset.df.shape}')

    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, num_workers=4)
    return dataloader


# TS-TCC

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        X_train = dataset[['psa_level0', 'psa_level1', 'psa_level2', 'psa_level3', 'psa_level4', 'psa_level5']]
        y_train = dataset['pros_cancer']

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        target = np.array(self.y_data.loc[index])
        x = self.x_data.loc[index].astype('float32')
        x = x.values
        return target, x, x, x

    def __len__(self):
        return self.len


def data_generator(data, config):

    data = normalize(data)
    print(data.head(5))

    train_df, val_df = generate_split(data)
    train_df, test_df = generate_split(train_df)

    train_dataset = Load_Dataset(train_df)
    valid_dataset = Load_Dataset(val_df)
    test_dataset = Load_Dataset(test_df)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                               shuffle=True, drop_last=config.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=config.batch_size,
                                               shuffle=False, drop_last=config.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader