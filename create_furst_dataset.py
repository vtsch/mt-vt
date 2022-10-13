import pandas as pd
import numpy as np

def reshape_psa_data_and_save():
    file_name = "data/psadata_furst_measurements.csv"
    df = pd.read_csv(file_name, header=0)

    # remove time from date
    df['ambiguous_date'] = df['ambiguous_date'].map(lambda x: x.split()[0])
    # print(df1.head(5))

    # merge date and measurement
    df['datapsa'] = df['ambiguous_date'] + [' '] + df['result_numeric'].astype(str)
    #print(df1.head(5))

    # put all measurements and dates in columns per patient
    df1 = (df.set_index(['ss_number_id', df.groupby('ss_number_id').cumcount()])['datapsa']
            .unstack(fill_value='')
            .add_prefix('datapsa_')
            .reset_index())
    #print(df1.head(5))

    # delete all columns after column 20
    df1 = df1.iloc[:, :20]

    # split datapsa_i into date and measurement at empty space for all columns
    for i in range(0, 19):
            df1['date_' + str(i)] = [x[:10] for x in df1['datapsa_' + str(i)]]
            df1['psa_' + str(i)] = [x[10:] for x in df1['datapsa_' + str(i)]]
            #delete the old column
            del df1['datapsa_' + str(i)]
    #print(df1.head(5))

    df1.to_csv("/tsd/p594/home/p594-vanessat/mt-vt/data/psadata_furst_measurements_restructured.csv", index=False)

def get_measurements():
    file_name = "data/psadata_furst_measurements_restructured.csv"
    df = pd.read_csv(file_name, header=0)
    # print("nr of rows measurements: ", len(df)) # 1'087'385
    # drop rows with less than 9 entries --> need to have more than 4 measurements
    df = df.dropna(axis=0, thresh=9)
    # print("nr of rows psa data: ", len(df)) # 411'205
    return df

def get_ages():
    file_name = "data/psadata_furst_age.csv"
    df = pd.read_csv(file_name, header=0)
    df['date_of_birth_15'] = df['date_of_birth_15'].map(lambda x: x.split()[0])
    # print("nr of rows ages: ", len(df)) # 3'771'007
    # drop rows where birthday is before 1979 --> no, otherwise only 3019 datapoints
    # df = df[df.date_of_birth_15 > '1979-01-01']
    return df

def get_labels():
    file_name = "data/psadata_furst_labels.csv"
    df = pd.read_csv(file_name, header=0)
    # print("nr of rows labels: ", len(df)) # 155'865
    # delete rows with 'missing' in npcc_risk_class_group_1 as missing = couldn't make diagnosis
    df = df[df.npcc_risk_class_group_1 != 'Missing']
    # encode risk classes as 1, 2, 3
    df['npcc_risk_class_group_1'] = df['npcc_risk_class_group_1'].map(lambda x: 1 if x == 'Localized' else x)
    df['npcc_risk_class_group_1'] = df['npcc_risk_class_group_1'].map(lambda x: 2 if x == 'Advanced' else x)
    df['npcc_risk_class_group_1'] = df['npcc_risk_class_group_1'].map(lambda x: 3 if x == 'Metastatic' else x)
    # print("nr of rows labels without missing: ", len(df)) # 118'698
    return df


if __name__ == "__main__":
    #reshape_psa_data_and_save()
    df_m = get_measurements()
    df_a = get_ages()
    df_l = get_labels()

    # merge dataframe psa levels and ages on id
    df = pd.merge(df_m, df_a, on='ss_number_id', how='inner')
    # print("nr of rows after merge psa and age: ", len(df)) # 411'205
    # check: print nr of rows without entry in date_of_birth_15
    # print("nr of rows without entry in date_of_birth_15: ", len(df[df.date_of_birth_15.isnull()])) # 0
    df = pd.merge(df, df_l, on='ss_number_id', how='left')
    # print("nr of rows after merge labels: ", len(df)) # 411'205
    # fill 0 for all patients without label
    df['npcc_risk_class_group_1'] = df['npcc_risk_class_group_1'].fillna(0)
    # create row cancer, add 0 if npcc_risk_class_group_1 == 0, else 1
    df['cancer'] = np.where(df['npcc_risk_class_group_1'] == 0, 0, 1)
    #print nr of patients with cancer
    print("nr of patients with cancer: ", len(df[df['cancer'] == 1])) # 79'053
    print("nr of patients without cancer: ", len(df[df['cancer'] == 0])) # 332'152

    # print nr of entries in each column without nan
    # print(df.count())

    print(df.head(5))

    # save the dataframe
    df.to_csv("/tsd/p594/home/p594-vanessat/mt-vt/data/psadata_furst.csv", index=False)
