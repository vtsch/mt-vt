import numpy as np
import pandas as pd
from bunch import Bunch
from typing import Tuple
from models.tstcc_TC import DataTransform
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import normalize
from sklearn.model_selection import train_test_split


class LoadPSADataset(Dataset):
    '''
    Parameters:
        config: config file
        data: dataframe
    Returns:
        dataset
    '''

    def __init__(self, config: Bunch, data: pd.DataFrame) -> None:
        '''
        Initialize dataset
        Args:
            config: config file
            data: dataframe
        '''

        self.df = data
        self.config = config

        y = data[config.classlabel]

        # data columns of PLCO dataset
        if config.dataset == "plco":
            X = data[['psa_level0', 'psa_level1', 'psa_level2',
                      'psa_level3', 'psa_level4', 'psa_level5']]

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
        # data columns of Furst dataset
        elif config.dataset == "furst":
            X = data[['psa_0', 'psa_1', 'psa_2', 'psa_3', 'psa_4', 'psa_5', 'psa_6', 
                        'psa_7', 'psa_8', 'psa_9', 'psa_10', 'psa_11', 'psa_12', 'psa_13', 
                        'psa_14', 'psa_15', 'psa_16', 'psa_17', 'psa_18', 'psa_19']]

            if self.config.pos_enc == "absolute_days":
                ts = data[['psa_absolute0', 'psa_absolute1', 'psa_absolute2', 'psa_absolute3', 
                            'psa_absolute4', 'psa_absolute5', 'psa_absolute6', 'psa_absolute7',
                            'psa_absolute8', 'psa_absolute9', 'psa_absolute10', 'psa_absolute11', 
                            'psa_absolute12', 'psa_absolute13', 'psa_absolute14', 'psa_absolute15', 
                            'psa_absolute16', 'psa_absolute17', 'psa_absolute18', 'psa_absolute19']]
            elif self.config.pos_enc == "delta_days":
                ts = data[['psa_delta0', 'psa_delta1', 'psa_delta2', 'psa_delta3', 'psa_delta4', 
                            'psa_delta5', 'psa_delta6', 'psa_delta7', 'psa_delta8', 'psa_delta9',
                            'psa_delta10', 'psa_delta11', 'psa_delta12', 'psa_delta13', 'psa_delta14', 
                            'psa_delta15', 'psa_delta16', 'psa_delta17', 'psa_delta18', 'psa_delta19']]

            elif self.config.pos_enc == "age_pos_enc":
                ts = data[['psa_age0', 'psa_age1', 'psa_age2', 'psa_age3', 'psa_age4', 
                            'psa_age5', 'psa_age6', 'psa_age7', 'psa_age8', 'psa_age9', 
                            'psa_age10', 'psa_age11', 'psa_age12', 'psa_age13', 'psa_age14', 
                            'psa_age15', 'psa_age16', 'psa_age17', 'psa_age18', 'psa_age19']]
            else:
                ts = pd.DataFrame(np.zeros((len(data), 20)))
        else:
            raise ValueError("Dataset not supported")

        # context data
        if self.config.context:
            #context = data[['bmi_curr', 'center', 'age', 'race7']]
            context_b = 'bmi_curr' if self.config.context_bmi else None
            context_c = 'center' if self.config.context_center else None
            context_a = 'age' if self.config.context_age else None
            context_indices = [context_b, context_c, context_a]
            context_indices = [i for i in context_indices if i is not None]
            context = data[context_indices]
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
            # make sure the Channels in second dim
            X_np = X_np.reshape(X_np.shape[0], 1, X_np.shape[1])
            X_np = torch.from_numpy(X_np)
            self.aug1, self.aug2 = DataTransform(X_np)

    def __getitem__(self, index: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Get item from dataset
        Args:
            index: index of item
        Returns:
            signal: PSA data
            target: class label
            tsindex: time series index
            context: context data

            if ts_tcc additionally:
                aug1: augmented data weak
                aug2: augmented data strong
        '''
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

    def __len__(self) -> int:
        '''
        Get length of dataset
        Returns:
            length of dataset
        '''
        return self.len


def generate_split(data: pd.DataFrame, config: Bunch) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Generate train and test split
    Args:
        data: dataset
        config: configuration
    Returns:
        train_df: train split
        test_df: test split
    '''
    train_df, test_df = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data[config.classlabel]
    )
    train_df, test_df = train_df.reset_index(
        drop=True), test_df.reset_index(drop=True)
    return train_df, test_df


def get_dataloader(config: Bunch, data: Dataset, phase: str) -> DataLoader:
    '''
    Parameters:
        config: config file
        data: dataframe
        phase: training, validation or test
    Returns:
        data generator
    '''
    data = normalize(data)

    train_df, test_df = generate_split(data, config)
    train_df, val_df = generate_split(train_df, config)

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
    print(df[config.classlabel].value_counts())

    dataloader = DataLoader(
        dataset=dataset, batch_size=config.batch_size, drop_last=True, num_workers=0)
    return dataloader


# TS-TCC
def data_generator_tstcc(data: Dataset, config: Bunch) -> DataLoader:
    '''
    Parameters:
        config: config file
        data: dataframe
    Returns:
        data generator
    '''
    data = normalize(data)

    train_df, val_df = generate_split(data, config)
    train_df, test_df = generate_split(train_df, config)

    train_dataset = LoadPSADataset(config, train_df)
    valid_dataset = LoadPSADataset(config, val_df)
    test_dataset = LoadPSADataset(config, test_df)
    print(f'train data shape: {train_dataset.df.shape}')
    print(f'valid data shape: {valid_dataset.df.shape}')
    print(f'test data shape: {test_dataset.df.shape}')

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                             shuffle=True, drop_last=True, num_workers=0)

    return train_loader, valid_loader, test_loader
