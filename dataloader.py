import numpy as np
import pandas as pd

from models.tstcc_TC import DataTransform

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

        if self.config.pos_enc == "absolute_days":
            ts = data[['psa_days0', 'psa_days1', 'psa_days2',
                       'psa_days3', 'psa_days4', 'psa_days5']]
        elif self.config.pos_enc == "delta_days":
            ts = data[['psa_delta0', 'psa_delta1', 'psa_delta2',
                       'psa_delta3', 'psa_delta4', 'psa_delta5']]
        elif self.config.pos_enc == "age_pos_enc":
            ts = data[['psa_age0', 'psa_age1', 'psa_age2',
                       'psa_age3', 'psa_age4', 'psa_age5']]
        else:
            ts = pd.DataFrame(np.zeros((len(data), 6)))
        
        if self.config.context:
            #context = data[['bmi_curr', 'center', 'age', 'race7']]
            context_b = 'bmi_curr' if self.config.context_bmi else None
            context_c = 'center' if self.config.context_center else None
            context_a = 'age' if self.config.context_age else None
            context_indices = [context_b, context_c, context_a]
            context_indices = [i for i in context_indices if i is not None]
            context = data[context_indices]
            #add 4 columns with zeros to ts if context is used with concat 
            #ts = pd.concat([ts, pd.DataFrame(np.zeros((ts.shape[0], self.config.context_count)))], axis=1)

        else: 
            context = data[[]]

        if isinstance(X, np.ndarray):
            self.x_data = torch.from_numpy(X)
            self.ts_data = torch.from_numpy(ts)
            self.y_data = torch.from_numpy(y).long()
            self.context_data = torch.from_numpy(context)
        else:
            self.x_data = X
            self.ts_data = ts
            self.y_data = y
            self.context_data = context

        self.len = X.shape[0]

        if self.config.experiment_name == "ts_tcc": 
            X_np = np.array(X)
            X_np = X_np.reshape(X_np.shape[0], 1, X_np.shape[1]) # make sure the Channels in second dim
            X_np = torch.from_numpy(X_np)
            self.aug1, self.aug2 = DataTransform(X_np, config)

    def __getitem__(self, index):
        target = np.array(self.y_data.loc[index])
        signal = self.x_data.loc[index].astype('float32')
        signal = signal.values
        tsindex = self.ts_data.loc[index].astype('float32')
        tsindex = tsindex.values
        context = self.context_data.loc[index].astype('float32')
        context = context.values
        #print("target", target, "tsindex", tsindex, "signal", signal, "context", context)

        if self.config.experiment_name == "ts_tcc":
            aug1 = self.aug1[index]
            aug2 = self.aug2[index]
            return signal, target, aug1, aug2, tsindex, context
        else:
            return signal, target, tsindex, context

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

    print(df['pros_cancer'].value_counts())

    dataloader = DataLoader(
        dataset=dataset, batch_size=config.batch_size, drop_last=True, num_workers=0)
    return dataloader


# TS-TCC
def data_generator_tstcc(data, config):

    data = normalize(data)

    train_df, val_df = generate_split(data)
    train_df, test_df = generate_split(train_df)

    train_dataset = LoadPSADataset(config, train_df)
    valid_dataset = LoadPSADataset(config, val_df)
    test_dataset = LoadPSADataset(config, test_df)
    print(f'train data shape: {train_dataset.df.shape}')
    print(f'valid data shape: {valid_dataset.df.shape}')
    print(f'test data shape: {test_dataset.df.shape}')

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size,
                              shuffle=False, drop_last=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                             shuffle=False, drop_last=True, num_workers=0)

    return train_loader, valid_loader, test_loader
