import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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


# ---- utils ----
def upsample_data(df_mitbih, n_clusters, sample_size):
    #select sample_size samples from each class
    df = pd.DataFrame()
    for i in range(n_clusters):
        #df = pd.concat([df, df_mitbih.loc[df_mitbih['class'] == i].sample(n=sample_size)])
        df = pd.concat([df, df_mitbih.loc[df_mitbih['pros_cancer'] == i].sample(n=sample_size, replace=True)])
    #shuffle rows of df and remove index column
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)  
    #print(df.head(5))

    return df

def normalize(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def generate_split(data):
    train_df, test_df = train_test_split(
        # data, test_size=0.15, random_state=42, stratify=data['class']
        data, test_size=0.2, random_state=42, stratify=data['pros_cancer']
    )
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    return train_df, test_df

# ---- for PSA data ----

def load_psa_df(df):
    # select columns with psa data, set threshold to have at least 5 measurements
    # psa_levels per year: 69-74
    # level from most recent test prior to diagnosis: 5
    # pros_cancer label: 4
    df = df.iloc[:, [69, 70, 71, 72, 73, 74, 5, 4]]
    df.dropna(thresh=4, inplace=True)
    df.fillna(0, inplace=True)
    #print("nr of labels")
    #print(df['pros_cancer'].value_counts())
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

def load_timesteps_df(df):
    # select columns with timesteps data
    # pclo_id: 44
    # pros_dx_psa_gap: time last psa measurement to diagnosis: 6
    # pros_exitdays: entry to pros cancer diagnosis or trial exit: 49
    # mortality_exitdays: days until mortality, last day known to be alive: 174
    # day of psa level mesaurements: 80-85
    # pros_cancer label: 4
    df = df.iloc[:, [80, 81, 82, 83, 84, 85, 4]]
    df.dropna(thresh=4, inplace=True)
    df.fillna(0, inplace=True)
    return df

def load_psa_and_timesteps_df(df):
    # psa_levels per year: 69-74
    # day of psa level mesaurements: 80-85
    # pros_cancer label: 4
    df = df.iloc[:, [69, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84, 85, 4]]
    df.dropna(thresh=8, inplace=True)
    df.fillna(0, inplace=True)
    return df

def load_psa_and_deltatime_df(df):
    # psa_levels per year: 69-74
    # day of psa level mesaurements: 80-85
    # pros_cancer label: 4
    df = df.iloc[:, [69, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84, 85, 4]]
    df.dropna(thresh=8, inplace=True)
    df.fillna(0, inplace=True)
    print(df.columns)
    #calculate deltatime between psa measurements
    df['psa_delta0'] = 0
    df['psa_delta1'] = df['psa_days1'] - df['psa_days0']
    df['psa_delta2'] = df['psa_days2'] - df['psa_days1']
    df['psa_delta3'] = df['psa_days3'] - df['psa_days2']
    df['psa_delta4'] = df['psa_days4'] - df['psa_days3']
    df['psa_delta5'] = df['psa_days5'] - df['psa_days4']

    df.drop(['psa_days0', 'psa_days1', 'psa_days2', 'psa_days3', 'psa_days4', 'psa_days5'], axis=1, inplace=True)
    df[df < 0] = 0
    print(df)
    return df

def load_psa_data_to_pd(file_name: str, config: dict) -> pd.DataFrame:
    '''
    Parameters:
        file_name: str
        config:dictionary with configuration parameters
    Returns:
        df_psa_ts: pd.DataFrame with psa values and timestep index and labels
    '''
    df_raw = pd.read_csv(file_name, header=0)
    if config.DELTATIMES:
        df = load_psa_and_deltatime_df(df_raw)
    else:
        df = load_psa_and_timesteps_df(df_raw)
        
    if config.upsample:
        df = upsample_data(df, n_clusters=config.n_clusters_real, sample_size=config.sample_size)

    return df


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