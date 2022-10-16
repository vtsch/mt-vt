#!/bin/sh

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc none  -tstcc_tm fine_tune -tstcc_dir '22-10-12_14-49-43' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-12_15-15-28' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-12_18-52-02' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-12_19-23-41' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-12_20-02-51' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-12_21-52-55' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-12_22-17-35' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-12_22-46-17' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-12_23-12-07' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-12_23-41-02' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_00-09-06' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_00-37-34' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_01-08-04' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_01-35-24' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_02-04-54' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_02-32-37' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_03-01-45' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_03-28-24' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_03-55-38' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_04-23-55' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_04-54-00' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_05-21-00' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_05-50-49' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_06-21-38' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_06-51-57' 	


python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 3 -pos_enc none  -tstcc_tm fine_tune -tstcc_dir '22-10-13_07-18-19' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 3 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-13_07-45-45' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 3 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-13_08-15-08' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 3 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-13_08-45-37' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 3 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-13_09-08-52' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_09-36-29' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_10-03-47' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_10-31-57' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_11-12-41' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_12-03-13' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_12-48-27' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_13-37-18' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_01-08-04' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_15-16-09' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-13_15-50-07' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_16-42-21' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_17-28-25' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_18-06-06' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_18-52-32' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_19-55-09' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_20-34-38' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_21-14-01' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_21-53-25' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_22-33-51' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-13_23-35-09' 


python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc none  -tstcc_tm fine_tune -tstcc_dir '22-10-14_00-33-33' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-14_01-24-19' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-14_02-17-59' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-14_03-16-11' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-14_04-08-21' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-14_05-01-43' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-14_05-48-57' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-14_06-35-35' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-14_07-18-25' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-14_08-02-21' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-14_08-35-42' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-14_09-47-17' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-14_10-45-35' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-14_12-27-39' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-14_13-53-24' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-14_15-08-08' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-14_21-05-09' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-14_21-34-06' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-14_22-10-02' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-14_22-43-21' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-14_23-14-43' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-14_23-42-04' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-15_00-09-35' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-15_00-36-04' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-15_01-00-11' 
