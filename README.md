# mt-vt

## Master Thesis - Deep Predictive Clustering of Irregular Time Series Data for Prostate Cancer Risk and Its Explainability

* branch "playground" for first experiments
* branch "simple_models" for baseline models
* branch "transformer" for first transformer implementation
* branch "ts-tcc" for TS-TCC architecture, from <https://github.com/emadeldeen24/TS-TCC>

## How to create Furst dataset

run the following SQL commands in the database

1. psa levels <br />
   SELECT ss_number_id, ambiguous_date, result_numeric FROM psaresults
   WHERE ambiguous_date is not null
   ORDER BY ambiguous_date
2. birthdays <br />
   SELECT ss_number_id, date_of_birth_15 FROM ss_numbers 
   WHERE date_of_birth_15 is not null
3. get labels <br />
   SELECT ss_number_id, npcc_risk_class_group_1, npcc_risk_class_group_2, npcc_risk_class_group_3 FROM kreftreg_data
4. run preprocess_furst.py
