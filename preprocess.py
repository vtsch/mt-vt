import numpy as np
import pandas as pd
from bunch import Bunch
from sklearn.preprocessing import MinMaxScaler

# ---- utils for data processing ----


def upsample_data(df: pd.DataFrame, config: Bunch) -> pd.DataFrame:
    '''
    Upsample data to balance classes
    Args:
        df (pd.DataFrame): dataframe with psa levels and labels
    Returns:
        df (pd.DataFrame): dataframe with balanced classes
    '''
    upsampled_data = pd.DataFrame()
    # select sample_size samples from each class
    for i in range(config.n_clusters_real):
        upsampled_data = pd.concat([upsampled_data, df.loc[df[config.classlabel] == i].sample(
            n=config.sample_size, replace=True)])
    # shuffle rows of df and remove index column
    upsampled_data = upsampled_data.sample(frac=1)
    upsampled_data = upsampled_data.reset_index(drop=True)
    return upsampled_data


def normalize(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Normalize data
    Args:
        data: dataframe with psa levels and labels
    Returns:
        data: dataframe with normalized psa levels and labels
    '''
    min_max_scaler = MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    df = pd.DataFrame(data_scaled, columns=data.columns)
    return df


# ---- load PSA data with features, positional encodings and contexts ----

def load_plco_df(df: pd.DataFrame) -> pd.DataFrame:
    # psa_levels per year: 69-74
    # pros_cancer label: 4
    df = df.iloc[:, [69, 70, 71, 72, 73, 74, 4, 44]]
    return df


def load_furst_df(df: pd.DataFrame) -> pd.DataFrame:
    # ss_nr_id: 0
    # psa levels: 2, 4, 6, 8, 10, 12, ... 40
    # psa days: 1, 5, 7, 9, 11, 13, ... 39
    # date_of_birth: 41
    # npcc_risk_class_group_1-3: 42, 43, 44
    # cancer: 45
    df = df.iloc[:, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                     26, 28, 30, 32, 34, 36, 38, 40, 41, 42, 43, 44, 45, 0]]
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

    # df = df.iloc[:, [144, 201, 205, 120, 44]] #bmicurr, center, age, race
    context_b = 144 if config.context_bmi else None
    context_c = 201 if config.context_center else None
    context_a = 205 if config.context_age else None
    indices = [context_b, context_c, context_a, 44]
    indices = [i for i in indices if i is not None]
    df = df.iloc[:, indices]
    return df


def load_psa_and_absolutedays_df(df: pd.DataFrame, config: Bunch) -> pd.DataFrame:
    if config.dataset == "plco":
        # psa_levels per year: 69-74
        # day of psa level mesaurements: 80-85
        # pros_cancer label: 4
        df = df.iloc[:, [69, 70, 71, 72, 73,
                         74, 80, 81, 82, 83, 84, 85, 4, 44]]
    elif config.dataset == "furst":
        # psa levels: 2, 4, 6, 8, 10, 12, ... 42
        # psa days: 1, 3, 5, 7, 9, 11, 13, ... 41
        # cancer label: 47
        earliest_date = df['date_0'].min()
        for i in range(0, 20):
            df['psa_absolute' + str(i)] = (df['date_' + str(i)] -
                                           earliest_date) / np.timedelta64(1, 'D')
            df.drop(['date_' + str(i)], axis=1, inplace=True)
    else:
        raise ValueError("Dataset not supported")
    return df


def load_psa_and_deltadays_df(df: pd.DataFrame, config: Bunch) -> pd.DataFrame:
    '''
    Load PSA data with features and delta days positional encoding
    Args:
        df (pd.DataFrame): raw psa dataframe 
    Returns:
        df (pd.DataFrame): dataframe with psa levels and delta days (days between psa measurements)
    '''
    # calculate deltatime between psa measurements and add to dataframe and delete measurement days
    if config.dataset == "plco":
        df = df.iloc[:, [69, 70, 71, 72, 73,
                         74, 80, 81, 82, 83, 84, 85, 4, 44]]
        df['psa_delta0'] = 0
        for i in range(1, 6):
            df['psa_delta' + str(i)] = df['psa_days' +
                                          str(i)] - df['psa_days' + str(i-1)]
        df.drop(['psa_days0', 'psa_days1', 'psa_days2', 'psa_days3',
                'psa_days4', 'psa_days5'], axis=1, inplace=True)

    elif config.dataset == "furst":
        df['psa_delta0'] = 0
        for i in range(1, 20):
            df['psa_delta' + str(i)] = (df['date_' + str(i)] -
                                        df['date_' + str(i-1)]) / np.timedelta64(1, 'D')
        for i in range(0, 20):
            df.drop(['date_' + str(i)], axis=1, inplace=True)
    else:
        raise ValueError("Dataset not supported")

    return df


def load_psa_and_age_df(df: pd.DataFrame, config: Bunch) -> pd.DataFrame:
    '''
    Load PSA data with features and age position encoding
    Args:
        df (pd.DataFrame): raw psa dataframe 
    Returns:
        df (pd.DataFrame): dataframe with psa levels and age at trial date
    '''
    if config.dataset == "plco":
        # psa_levels per year: 69-74
        # age at trial entry: 205
        # pros_cancer label: 4
        df = df.iloc[:, [69, 70, 71, 72, 73, 74, 205, 4, 44]]
        # calculate age at psa measurement
        df['psa_age0'] = df['age']
        for i in range(1, 6):
            df['psa_age' + str(i)] = df['age'] + i

        df.drop(['age'], axis=1, inplace=True)

    elif config.dataset == "furst":
        for i in range(0, 20):
            df['psa_age' + str(i)] = (df['date_' + str(i)] -
                                      df['date_of_birth_15']) / np.timedelta64(1, 'Y')
            df.drop(['date_' + str(i)], axis=1, inplace=True)
    else:
        raise ValueError("Dataset not supported")

    return df


def load_psa_data_to_pd(file_name: str, config: dict) -> pd.DataFrame:
    '''
    Load PSA data from file, select PSA measurements, position encodings and context
    Args:
        file_name: str
        config: dictionary with configuration parameters
    Returns:
        df_psa_ts: pd.DataFrame with psa values and position encodings and labels
    '''
    # read data
    df_raw = pd.read_csv(file_name, header=0)

    # convert dates to datetime in furst dataset
    if config.dataset == 'furst':
        for i in range(0, 20):
            df_raw['date_' + str(i)] = pd.to_datetime(
                df_raw['date_' + str(i)], format='%Y-%m-%d')
        df_raw['date_of_birth_15'] = pd.to_datetime(
            df_raw['date_of_birth_15'], format='%Y-%m-%d')

    # load positional encodings
    if config.pos_enc == "absolute_days":
        df = load_psa_and_absolutedays_df(df_raw, config)
    elif config.pos_enc == "delta_days":
        df = load_psa_and_deltadays_df(df_raw, config)
    elif config.pos_enc == "age_pos_enc":
        df = load_psa_and_age_df(df_raw, config)
    else:
        if config.dataset == "plco":
            df = load_plco_df(df_raw)
        elif config.dataset == "furst":
            df = load_furst_df(df_raw)
        else:
            raise ValueError("Dataset not supported")

    # load context
    if config.context and config.dataset == "plco":
        df_context = create_context_df(df_raw, config)
        df = pd.merge(df, df_context, on='plco_id', how='inner')
    else:
        print("No context data available")

    # final cleanup for models
    if config.dataset == "plco":
        # drop rows that have nan in psa levels
        if df.iloc[:, :6].isnull().values.any():
            df.dropna(axis=0, inplace=True)
        df.drop(['plco_id'], axis=1, inplace=True)
        # fill nan values with -1
        df.fillna(0, inplace=True)

    elif config.dataset == "furst":
        df.drop(['ss_number_id'], axis=1, inplace=True)
        df.drop(['date_of_birth_15'], axis=1, inplace=True)
        df.drop(['npcc_risk_class_group_1'], axis=1, inplace=True)
        df.drop(['npcc_risk_class_group_2'], axis=1, inplace=True)
        df.drop(['npcc_risk_class_group_3'], axis=1, inplace=True)
        # fill nan values with -1
        df.fillna(-1, inplace=True)
    else:
        raise ValueError("Dataset not supported")

    if config.upsample:
        df_u = upsample_data(df, config)
        print("class distribution upsampeled:\n",
              df_u[config.classlabel].value_counts())
    else:
        df_u = df
        print("class distribution:\n", df[config.classlabel].value_counts())

    return df, df_u
