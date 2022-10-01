import numpy as np
import pandas as pd
from bunch import Bunch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---- utils ----
def upsample_data(df: pd.DataFrame, config: Bunch) -> pd.DataFrame:
    #select sample_size samples from each class
    upsampled_data = pd.DataFrame()
    for i in range(config.n_clusters_real):
        upsampled_data = pd.concat([upsampled_data, df.loc[df['pros_cancer'] == i].sample(n=config.sample_size, replace=True)])
    #shuffle rows of df and remove index column
    upsampled_data = upsampled_data.sample(frac=1)
    upsampled_data = upsampled_data.reset_index(drop=True)  
    return upsampled_data

def normalize(data: pd.DataFrame) -> pd.DataFrame:
    #x = df.values #returns a numpy array
    min_max_scaler = MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    df = pd.DataFrame(data_scaled, columns = data.columns)
    return df

def generate_split(data: pd.DataFrame) -> Bunch:
    train_df, test_df = train_test_split(
        # data, test_size=0.15, random_state=42, stratify=data['class']
        data, test_size=0.2, random_state=42, stratify=data['pros_cancer']
    )
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    return train_df, test_df


# ---- load PSA data with features, positional encodings and contexts ----

def load_plco_df(df: pd.DataFrame) -> pd.DataFrame:
    # select columns with psa data, set threshold to have at least 5 measurements
    # psa_levels per year: 69-74
    # pros_cancer label: 4
    df = df.iloc[:, [69, 70, 71, 72, 73, 74, 4, 44]]
    return df

def load_furst_df(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name, header=0)
    # select columns with psa data, set threshold to have at least 5 measurements
    # ss_nr_id =0
    # psa levels: 2, 4, 6, 8, 10, 12, ... 42
    # label: 44
    # in column 44, change all "localized" to 1, "advanced" to 2, "metastatic" to 3, "missing"  to 0 in column
    df.replace(to_replace="localized", value=1, inplace=True)
    df.replace(to_replace="advanced", value=2, inplace=True)
    df.replace(to_replace="metastatic", value=3, inplace=True)
    df.replace(to_replace="missing", value=0, inplace=True)

    df = df.iloc[:, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]]
    #df.dropna(thresh=4, inplace=True)
    if df.iloc[:, :8].isnull().values.any():
        df.dropna(axis=0, inplace=True)
        
    print("class distribution: \n", df['npcc_risk_class_group_1'].value_counts())
    df.fillna(0, inplace=True)
    print(df.head(10))
    return df


def create_context_df(df: pd.DataFrame, config: Bunch) -> pd.DataFrame:
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
    # CHOOSE WHICH CONTEXT TO ADD:

    #df = df.iloc[:, [144, 201, 205, 120, 44]] #bmicurr, center, age, race
    context_b = 144 if config.context_bmi else None
    context_c = 201 if config.context_center else None
    context_a = 205 if config.context_age else None
    indices = [context_b, context_c, context_a, 44]
    indices = [i for i in indices if i is not None]
    df = df.iloc[:, indices]
    return df

def load_psa_and_absolutedays_df(df: pd.DataFrame) -> pd.DataFrame:
    # psa_levels per year: 69-74
    # day of psa level mesaurements: 80-85
    # pros_cancer label: 4
    df = df.iloc[:, [69, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84, 85, 4, 44]]
    return df

def load_psa_and_deltadays_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Parameters:
        df (pd.DataFrame): raw psa dataframe 
    Returns:
        df (pd.DataFrame): dataframe with psa levels and delta days (days between psa measurements)
    '''
    # psa_levels per year: 69-74
    # day of psa level mesaurements: 80-85
    # pros_cancer label: 4
    df = df.iloc[:, [69, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84, 85, 4, 44]]
    #calculate deltatime between psa measurements
    df['psa_delta0'] = 0
    for i in range(1, 6):
        df['psa_delta' + str(i)] = df['psa_days' + str(i)] - df['psa_days' + str(i-1)]

    df.drop(['psa_days0', 'psa_days1', 'psa_days2', 'psa_days3', 'psa_days4', 'psa_days5'], axis=1, inplace=True)
    return df

def load_psa_and_age_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Parameters:
        df (pd.DataFrame): raw psa dataframe 
    Returns:
        df (pd.DataFrame): dataframe with psa levels and age at trial date
    '''
    # psa_levels per year: 69-74
    # age at trial entry: 205
    # pros_cancer label: 4
    df = df.iloc[:, [69, 70, 71, 72, 73, 74, 205, 4, 44]]
    # calculate age at psa measurement
    df['psa_age0'] = df['age']
    for i in range(1, 6):
        df['psa_age' + str(i)] = df['age'] + i
    
    df.drop(['age'], axis=1, inplace=True)
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

    if config.pos_enc == "absolute_days":
        df = load_psa_and_absolutedays_df(df_raw)
    elif config.pos_enc == "delta_days":
        df = load_psa_and_deltadays_df(df_raw)
    elif config.pos_enc == "age_pos_enc":
        df = load_psa_and_age_df(df_raw)
    else:
        df = load_plco_df(df_raw)
    
    if config.context:
        df_context = create_context_df(df_raw, config)
        df = pd.merge(df, df_context, on='plco_id', how='inner')

    if config.upsample:
        df = upsample_data(df, config)
    
    # drop rows that have nan in psa levels
    if df.iloc[:, :6].isnull().values.any():
        df.dropna(axis=0, inplace=True)

    print("class distribution: \n", df['pros_cancer'].value_counts())

    df.drop(['plco_id'], axis=1, inplace=True)
    #fill nan values with -1
    df.fillna(-1, inplace=True)
    #df[df < 0] = 0

    return df

    


