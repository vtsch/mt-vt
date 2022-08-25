import numpy as np
import pandas as pd

from models.TC import DataTransform

import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import generate_split, normalize


class LoadPSADataset(Dataset):
    def __init__(self, config, data):

        self.df = data
        self.config = config

        X = data[['psa_level0', 'psa_level1', 'psa_level2',
                  'psa_level3', 'psa_level4', 'psa_level5']]
        y = data['pros_cancer']

        if self.config.DELTATIMES:
            ts = data[['psa_delta0', 'psa_delta1', 'psa_delta2',
                       'psa_delta3', 'psa_delta4', 'psa_delta5']]
        else:
            ts = data[['psa_days0', 'psa_days1', 'psa_days2',
                       'psa_days3', 'psa_days4', 'psa_days5']]

        if isinstance(X, np.ndarray):
            self.x_data = torch.from_numpy(X)
            self.ts_data = torch.from_numpy(ts)
            self.y_data = torch.from_numpy(y).long()
        else:
            self.x_data = X
            self.ts_data = ts
            self.y_data = y

        self.len = X.shape[0]
    
        if config.tstcc_training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            X = np.array(X)
            X = X.reshape(X.shape[0], -1, 1) # make sure the Channels in second dim
            x_d = torch.from_numpy(X)
            self.aug1, self.aug2 = DataTransform(x_d, config)

    def __getitem__(self, index):
        target = np.array(self.y_data.loc[index])
        signal = self.x_data.loc[index].astype('float32')
        signal = signal.values
        tsindex = self.ts_data.loc[index].astype('float32')
        tsindex = tsindex.values
        #print("target", target, "tsindex", tsindex, "signal", signal)

        if self.config.MOD_TSTCC:
            if self.config.tstcc_training_mode == "self_supervised":
                aug1 = self.aug1[index].reshape(self.config.ts_length)
                aug2 = self.aug2[index].reshape(self.config.ts_length)
                return signal, target, aug1, aug2
            else:
                return signal, target, signal, signal
        else:
            return signal, target, tsindex

    def __len__(self):
        return self.len


def get_dataloader(config, data, phase: str) -> DataLoader:
    '''
    Parameters:
        config: config file
        data: dataframe
        phase: training, validation or test
    Returns:
        data generator
    '''
    data = normalize(data)

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

    dataset = LoadPSADataset(config, df)
    print(f'{phase} data shape: {dataset.df.shape}')

    dataloader = DataLoader(
        dataset=dataset, batch_size=config.batch_size, drop_last=True, num_workers=0)
    return dataloader


# TS-TCC
def data_generator(data, config):

    data = normalize(data)

    train_df, val_df = generate_split(data)
    train_df, test_df = generate_split(train_df)

    train_dataset = LoadPSADataset(config, train_df)
    valid_dataset = LoadPSADataset(config, val_df)
    test_dataset = LoadPSADataset(config, test_df)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size,
                              shuffle=False, drop_last=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                             shuffle=False, drop_last=False, num_workers=0)

    return train_loader, valid_loader, test_loader
