#!/bin/sh

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc none  -tstcc_tm fine_tune -tstcc_dir '22-10-06_00-26-40' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-06_00-46-16' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-06_01-02-04' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-06_01-20-19' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-10-06_01-38-13' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-06_01-58-05' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-06_02-11-58' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-06_02-22-10' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-06_02-28-16' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-06_02-35-05' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-06_02-42-01' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-06_02-48-39' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-06_02-55-29' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-06_03-03-49' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-06_03-11-27' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-06_03-18-51' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-06_03-26-26' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-06_03-33-37' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-06_03-40-28' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-06_03-48-09' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-06_03-55-04' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-06_04-02-09' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-06_04-08-39' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-06_04-15-00' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-06_04-20-53' 	
