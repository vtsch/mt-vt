import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


# ---- for ECG data ----
def create_irregular_ts(data):
    # random boolean mask for which values will be changed
    replace_rate = 0.3
    mask = np.random.choice([0, 1], size=data.shape, p=((1-replace_rate),replace_rate)).astype(np.bool)

    # random matrix the same shape
    r = np.zeros(shape=data.shape)

    # use  mask to replace values in input array
    data[mask] = r[mask]

    return data

def load_ecg_data_to_pd(file_name_train, file_name_test):
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

# ---- for PSA data ----

def load_psa_data_to_pd(file_name):
    df_raw = pd.read_csv(file_name, header=0)
    return df_raw

def create_psa_df(df):
    # select columns with psa data, set threshold to have at least 5 measurements
    # psa_levels per year: 69-74
    # level from most recent test prior to diagnosis: 5
    # pros_cancer label: 4
    df = df.iloc[:, [69, 70, 71, 72, 73, 74, 5, 4]]
    df.dropna(thresh=4, inplace=True)
    df.fillna(0, inplace=True)
    print("nr of labels")
    print(df['pros_cancer'].value_counts())
    return df

def create_gleas_df(df):
    # select columns with gleas data
    # pclo_id: 44
    # gleason score, gleason biopsy, gleason prostatectomy, source of score (1=p, 2=b)
    # pros_cancer label: 4
    df = df.iloc[:, [44, 23, 24, 25, 26, 4]]
    df.dropna(inplace=True)
    return df

def create_context_df(df):
    # select columns with demographic data
    # pclo_id: 44
    # bmi20, bmi50, bmicurr, height: 142, 143, 144, 149
    # mortality_age: 170
    # ph_any_trial, had any personal history of cancer before (should be 0, 1 yes, 9 unknown): 190
    # pros_cancer_first: is pros cancer first diagnosed cancer (should be 1): 39
    # center (study center): 201
    # rnd year: year of trial entry: 202
    # age (at trial entry, computed rom date of birth): 205 
    # agelevel: age categorization (0: <59, 1: 60-64, 2: 65-59, 3: >70): 206
    # pros_exitage: age at exit for first cancer diagnosis or age at trial exit otherwise: 38
    # race: 120
    # pros_cancer: 4
    df = df.iloc[:, [44, 142, 143, 144, 149, 170, 190, 39, 201, 205, 206, 38, 120, 4]]
    return df

def create_timesteps_df(df):
    # select columns with timesteps data
    # pclo_id: 44
    # pros_dx_psa_gap: time last psa measurement to diagnosis: 6
    # pros_exitdays: entry to pros cancer diagnosis or trial exit: 49
    # mortality_exitdays: days until mortality, last day known to be alive: 174
    # day of psa level mesaurements: 80-85
    df = df.iloc[:, [44, 6, 49, 174, 80, 81, 82, 83, 84, 85]]
    return df


# ---- Dataloader ----

class MyDataset(Dataset):

    def __init__(self, df):
        self.df = df
        self.data_columns = self.df.columns[:-2].tolist()

    def __getitem__(self, idx):
        signal = self.df.loc[idx, self.data_columns].astype('float32')
        signal = torch.tensor(np.array([signal.values]))         
        # target = torch.LongTensor(np.array(self.df.loc[idx, 'class']))
        target = torch.tensor(np.array(self.df.loc[idx, 'pros_cancer']))
        return signal, target

    def __len__(self):
        return len(self.df)

def generate_train_test(data):
    train_df, val_df = train_test_split(
        # data, test_size=0.15, random_state=42, stratify=data['class']
        data, test_size=0.15, random_state=42, stratify=data['pros_cancer']
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
    print(f'{phase} data shape: {df.shape}')
    dataset = MyDataset(df)
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
    test_data = test_data.reset_index(drop=True)
    dataset = MyDataset(test_data)
    print(f'test data shape: {dataset.df.shape}')
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)
    return dataloader