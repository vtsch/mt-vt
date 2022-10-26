#!/bin/sh

python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc none  -tstcc_tm fine_tune -tstcc_dir '22-10-16_17-38-40' 
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-10-17_22-58-07' 
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-10-24_10-26-48' 
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-25_02-23-16' 
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-10-25_09-15-48' 

python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc none  -tstcc_tm train_linear -tstcc_dir '22-10-16_17-38-40' 
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-17_22-58-07' 
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-24_10-26-48' 
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-25_02-23-16'
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-25_09-15-48' 

