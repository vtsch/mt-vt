import pandas as pd
import numpy as np

def reshape_psa_data_and_save():
    file_name = "data/psadata_furst_raw.csv"
    df = pd.read_csv(file_name, header=0)

    # remove time from date
    df['sampling_date'] = df['sampling_date'].map(lambda x: x.split()[0])
    # print(df1.head(5))

    # merge date and measurement
    df['datapsa'] = df['sampling_date'] + [' '] + df['result_numeric'].astype(str)
    #print(df1.head(5))

    # put all measurements and dates in columns per patient
    df1 = (df.set_index(['ss_number_id', df.groupby('ss_number_id').cumcount()])['datapsa']
            .unstack(fill_value='')
            .add_prefix('datapsa_')
            .reset_index())
    #print(df1.head(5))

    # delete all columns after column 21
    df1 = df1.iloc[:, :22]

    # split datapsa_i into date and measurement at empty space for all columns
    for i in range(0, 21):
            df1['date_' + str(i)] = [x[:10] for x in df1['datapsa_' + str(i)]]
            df1['psa_' + str(i)] = [x[10:] for x in df1['datapsa_' + str(i)]]
            #delete the old column
            del df1['datapsa_' + str(i)]
    #print(df1.head(5))

    df1.to_csv("/tsd/p594/home/p594-vanessat/mt-vt/data/psadata_furst_measurements.csv", index=False)

def get_measurements():
    file_name = "data/psadata_furst_measurements.csv"
    df = pd.read_csv(file_name, header=0)
    # drop rows with less than 9 entries --> need to have more than 4 measurements
    df = df.dropna(axis=0, thresh=9)
    print("nr of rows psa data: ", len(df)) # 912'111 --> with threshold 514'002
    return df

def get_ages():
    file_name = "data/psadata_furst_age.csv"
    df = pd.read_csv(file_name, header=0)
    df['date_of_birth_15'] = df['date_of_birth_15'].map(lambda x: x.split()[0])
    print("nr of rows ages: ", len(df)) # 1'879'007
    return df

def get_labels():
    file_name = "data/psadata_furst_labels.csv"
    df = pd.read_csv(file_name, header=0)
    print("nr of rows labels: ", len(df)) #155'865
    return df


if __name__ == "__main__":
    # reshape_psa_data_and_save()
    df_m = get_measurements()
    df_a = get_ages()
    df_l = get_labels()

    # merge dataframe psa levels and ages on id
    df = pd.merge(df_m, df_a, on='ss_number_id')
    print("nr of rows after merge psa and age: ", len(df)) # 256'834
    df = pd.merge(df, df_l, on='ss_number_id', how='left')
    print("nr of rows after merge labels: ", len(df))
    print(df.head(5))

    # save the dataframe
    df.to_csv("/tsd/p594/home/p594-vanessat/mt-vt/data/psadata_furst.csv", index=False)
