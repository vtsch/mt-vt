# Clustering of PSA Data for Prostate Cancer Risk Classification and Its Explainability

Master Thesis - Vanessa A. Tschichold - ETH Zürich & NTNU

Here you find first the instructions on how to download and prepare the datasets, how to run the code and how the folder structure for the results is organized.

## Datasets

How to access and prepare the two datasets used:

### PLCO

Apply for access and then download the database from the Cancer Data Access System of the National Cancer Institute here: <https://cdas.cancer.gov/datasets/plco/20/>. Move the pros_data_mar22_d032222.csv file in the /data folder.

### Fürst

Only applicable if you are allowed to access the data and have a Norwegian MinID. 

- Log in into the VM and the pgAdmin Database.
- Run the following SQL commands in the database:
  
  1. get psa measurements
     SELECT ss_number_id, ambiguous_date, result_numeric FROM psaresults
     WHERE ambiguous_date is not null
     ORDER BY ambiguous_date

  2. get birthdays
     SELECT ss_number_id, date_of_birth_15 FROM ss_numbers
     WHERE date_of_birth_15 is not null

  3. get labels
     SELECT ss_number_id, npcc_risk_class_group_1, npcc_risk_class_group_2, npcc_risk_class_group_3 FROM kreftreg_data

- Save the created tables as csv in the /data folder as _psadata_furst_measurements.csv_, _psadata_furst_age.csv_, and _psadata_furst_labels.csv_ respectively
- Run the script: run `create_furst_dataset.py` to finish the preprocessing

## Running the code

- Create a python3 virtual environment
- Install all requirements in requirements.txt: `pip3 install -r requirements.txt`
- Load the data into /data/ folder
- To run all experiments of a model, run the respective script in the /scripts folder
- To run a single experiment, run python3 main.py with the following command line arguments
  - `-c` "configfile": change to configs/config_c_{insert letter} : f, all, a, b, c for false, all, age, BMI or center or make your own config file 
  - `-exp` "experiment name":  raw_data, simple_ae, lstm, cnn, simple_transformer, ts_tcc
  - `-n_clusters` "n": specify how many clusters dtw k-means should take (2, 3, 4) 
  - `-pos_enc` "position encoding" --> none, absolute_days, delta_days, age_pos_enc, learnable_pos_enc
  
- To run for TS-TCC also specify
  - `-tstcc_tm` "trainingmode": supervised, self_supervised, fine_tune, train_linear
  - for fine-tune and train linear, first pretrain with mode self_supervised and also add:
  - `-tstcc_dir` 'yy-mm-dd_hh-mm-ss': being the last created directory in the self-supervised saved models folder). i.e. for example: `python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-14_23-14-43'`
- note that the parameters in the `build_comet_logger` method in utils.py need to be changed to track the experiments in another comet project

- all models and graphs will be saved in the saved_models folder

## Folder Structure of the Results
The plots and the calculcated scores can be found in the respective model directory in _/saved_models_. The structure is the following: _experiment name/tstcc experiment name/position encoding/context_.
The ts-tcc experiment name is supervised per default (i.e. for the baselines) and the context vectors folder names are none, all, a, b or c.

The folders with the results (scores, plots for explainability and saved models) have the name of the daytime of the experiment.
For the experiments on the original dataset, the earliest folder is always _n\_clusters = 2_ and the latest _n\_clusters = 4_. 
The results on the balanced datasets are one level further down in the folder _/bal_.



## Links

* TS-TCC from <https://github.com/emadeldeen24/TS-TCC>
* CNN and KNN from <https://github.com/Wickstrom/MixupContrastiveLearning>
* Simpe Transformer inspired from <https://github.com/Sarunas-Girdenas/transformer_for_time_series>

### old - Overview of the branches

* branch "playground" for first experiments on ECG data
* branch "simple_models" for baseline models
* branch "transformer" for first transformer implementation
* branch "tstcc" for TS-TCC architecture
* branch "pos_enc" for adding the positional encodings
* branch "add_context" to implement preprocessing and loading of the context vectors
* branch "main" - main branch, finished architecture
