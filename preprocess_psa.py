import pandas as pd

file_name = "data/pros_data_mar22_d032222.csv"

def load_raw_data_to_pd(file_name):
    df_raw = pd.read_csv(file_name, header=0)
    return df_raw


def create_psa_df(df):
    # select columns with psa data, set threshold to have at least 5 measurements
    # pclo_id: 44
    # psa_levels per year: 69-74
    # level from most recent test prior to diagnosis: 5
    # pros_cancer label: 4
    df = df.iloc[:, [44, 69, 70, 71, 72, 73, 74, 5, 4]]
    df.dropna(thresh=4, inplace=True)
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

if __name__ == '__main__':
    file_name = "data/pros_data_mar22_d032222.csv"
    df_raw = load_raw_data_to_pd(file_name)
    df_psa = create_psa_df(df_raw)
    df_gleas = create_gleas_df(df_raw)
    df_context = create_context_df(df_raw)
    df_timesteps = create_timesteps_df(df_raw)
    print(df_psa.head(5))
