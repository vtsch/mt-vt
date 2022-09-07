import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
def upsample_data(df, config):
    #select sample_size samples from each class
    upsampled_data = pd.DataFrame()
    for i in range(config.n_clusters_real):
        upsampled_data = pd.concat([upsampled_data, df.loc[df['pros_cancer'] == i].sample(n=config.sample_size, replace=True)])
    #print nr of values in each class
    #print(df['pros_cancer'].value_counts())
    #shuffle rows of df and remove index column
    upsampled_data = upsampled_data.sample(frac=1)
    upsampled_data = upsampled_data.reset_index(drop=True)  
    return upsampled_data

def normalize(data):
    #x = df.values #returns a numpy array
    min_max_scaler = MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    df = pd.DataFrame(data_scaled, columns = data.columns)
    return df

def generate_split(data):
    train_df, test_df = train_test_split(
        # data, test_size=0.15, random_state=42, stratify=data['class']
        data, test_size=0.2, random_state=42, stratify=data['pros_cancer']
    )
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    return train_df, test_df

# ---- for PSA data ----

def load_psa_df(file_name):
    df = pd.read_csv(file_name, header=0)
    # select columns with psa data, set threshold to have at least 5 measurements
    # psa_levels per year: 69-74
    # level from most recent test prior to diagnosis: 5
    # pros_cancer label: 4
    df = df.iloc[:, [69, 70, 71, 72, 73, 74, 5, 4]]
    df.dropna(thresh=4, inplace=True)
    df.fillna(0, inplace=True)
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
    df = df.iloc[:, [144, 149, 205, 120, 44]]
    return df

def load_timesteps_df(df):
    # select columns with timesteps data
    # pclo_id: 44
    # pros_dx_psa_gap: time last psa measurement to diagnosis: 6
    # pros_exitdays: entry to pros cancer diagnosis or trial exit: 49
    # mortality_exitdays: days until mortality, last day known to be alive: 174
    # day of psa level mesaurements: 80-85
    # pros_cancer label: 4
    df = df.iloc[:, [44, 80, 81, 82, 83, 84, 85, 4]]
    df.dropna(thresh=4, inplace=True)
    df.fillna(0, inplace=True)
    return df

def load_psa_and_timesteps_df(df):
    # psa_levels per year: 69-74
    # day of psa level mesaurements: 80-85
    # pros_cancer label: 4
    df = df.iloc[:, [69, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84, 85, 4, 44]]
    df = df.dropna(thresh=8)
    df.fillna(0, inplace=True)
    return df

def load_psa_and_deltatime_df(df):
    # psa_levels per year: 69-74
    # day of psa level mesaurements: 80-85
    # pros_cancer label: 4
    df = df.iloc[:, [69, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84, 85, 4]]
    df.dropna(axis=0, thresh=8, inplace=True)
    df.fillna(value=0, inplace=True)
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
        df = upsample_data(df, config)
    
    df.drop(['plco_id'], axis=1, inplace=True)

    return df

def load_psa_data_and_context_to_pd(file_name: str, config: dict) -> pd.DataFrame:
    '''
    Parameters:
        file_name: str
        config:dictionary with configuration parameters
    Returns:
        df_psa_ts: pd.DataFrame with psa values and timestep index and labels
    '''
    df_raw = pd.read_csv(file_name, header=0)

    df_psa = load_psa_and_timesteps_df(df_raw)
    df_context = create_context_df(df_raw)

    #merge psa and context dataframes
    df = pd.merge(df_psa, df_context, on='plco_id', how='inner')

    if config.upsample:
        df = upsample_data(df, config)
    
    #delete plco_id column
    df.drop(['plco_id'], axis=1, inplace=True)
    df.fillna(value=-1, inplace=True)

    return df
    


