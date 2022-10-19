# Master Thesis - Clustering of PSA Data for Prostate Cancer Risk Classification and Its Explainability

## Datasets

How to access and prepare the two datasets used:

### PLCO

Apply for access and then download the database from the Cancer Data Access System of the National Cancer Institute here: <https://cdas.cancer.gov/datasets/plco/20/>

### Fürst

Only applicable if you are allowed to access the data and have a Norwegian MinID. Log in into the VM and the pgAdmin Database.
Then run the following SQL commands in the database:

1. get psa measurements
   SELECT ss_number_id, ambiguous_date, result_numeric FROM psaresults
   WHERE ambiguous_date is not null
   ORDER BY ambiguous_date

2. get birthdays
   SELECT ss_number_id, date_of_birth_15 FROM ss_numbers
   WHERE date_of_birth_15 is not null

3. get labels
   SELECT ss_number_id, npcc_risk_class_group_1, npcc_risk_class_group_2, npcc_risk_class_group_3 FROM kreftreg_data

Save the downloaded tables as csv in the /data/ folder as _psadata_furst_measurements.csv_, _psadata_furst_age.csv_, and _psadata_furst_labels.csv_ respectively. 

Then run the script: run create_furst_dataset.py to finish the preprocessing.

## Running the code

- Create a python3 virtual environment
- Install all requirements in requirements.txt: `pip3 install -r requirements.txt`
- Load the data into /data/ folder
- To run all experiments of a model, run the respective script in the /scripts folder
- To run a single experiment, run python3 main.py with the following arguments
  - `-c` "configfile": change to configs/config_c_{insert letter} : f, all, a, b, c for false, all, age, BMI or center or make your own config file 
  - `-exp` "experiment name":  raw_data, simple_ae, lstm, cnn, simple_transformer, ts_tcc
  - `-n_clusters` "n": specify how many clusters dtw k-means should take (2, 3, 4) 
  - `-pos_enc` "position encoding" --> none, absolute_days, delta_days, age_pos_enc, learnable_pos_enc
- To run for TS-TCC also specify
  - `-tstcc_tm` "trainingmode": supervised, self_supervised, fine_tune, train_linear
  - for fine-tune and train linear, first pretrain with mode self_supervised and also add:
  - `-tstcc_dir` 'yy-mm-dd_hh-mm-ss': being the last created directory in the self-supervised saved models folder). i.e. for example: `python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-14_23-14-43'`






### Overview of the branches

For tracing back:

* branch "playground" for first experiments on ECG data
* branch "simple_models" for baseline models
* branch "transformer" for first transformer implementation
* branch "tstcc" for TS-TCC architecture, from <https://github.com/emadeldeen24/TS-TCC>
* branch "pos_enc" for adding the positional encodings
* branch "add_context" to implement preprocessing and loading of the context vectors
* branch "main" - main branch, finished architecture
