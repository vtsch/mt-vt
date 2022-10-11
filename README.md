# mt-vt

## Master Thesis - Deep Predictive Clustering of Irregular Time Series Data for Prostate Cancer Risk and Its Explainability

## Datasets

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

To run the code, install all requirements

##  Random

For tracing back:

* branch "playground" for first experiments on ECG data
* branch "simple_models" for baseline models
* branch "transformer" for first transformer implementation
* branch "tstcc" for TS-TCC architecture, from <https://github.com/emadeldeen24/TS-TCC>
* branch "pos_enc" for adding the positional encodings
* branch "add_context" to implement preprocessing and loading of the context vectors
* branch "main" - main branch, finished architecture
