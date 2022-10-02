#!/bin/sh
python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc none  -tstcc_tm fine_tune -tstcc_dir '22-09-29_10-28-48f' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-09-29_11-04-24all' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-09-29_13-18-03a' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-09-29_16-24-45b' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm fine_tune -tstcc_dir '22-09-29_14-51-12c' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-09-29_10-28-08f' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-09-29_11-58-03all' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-09-29_13-21-45a' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-09-29_16-42-47b' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm fine_tune -tstcc_dir '22-09-29_14-51-49c' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-09-29_10-46-54f' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-09-29_11-58-28all' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-09-29_13-42-10a' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-09-29_16-49-19b' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm fine_tune -tstcc_dir '22-09-29_15-17-33c' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-09-29_11-25-21f' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-09-29_12-49-01all' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-09-29_14-06-54a' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-09-29_17-10-17b' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-09-29_15-38-03c' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-09-29_11-24-04f' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-09-29_12-51-08all' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-09-29_14-04-11a' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-09-29_17-11-09b' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm fine_tune -tstcc_dir '22-09-29_15-38-25c' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc none  -tstcc_tm train_linear -tstcc_dir '22-09-29_10-28-48f' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-09-29_11-04-24all' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-09-29_13-18-03a' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-09-29_16-24-45b' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-09-29_14-51-12c' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-09-29_10-28-08f' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-09-29_11-58-03all' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-09-29_13-21-45a' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-09-29_16-42-47b' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-09-29_14-51-49c' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-09-29_10-46-54f' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-09-29_11-58-28all' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-09-29_13-42-10a' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-09-29_16-49-19b' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-09-29_15-17-33c' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-09-29_11-25-21f' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-09-29_12-49-01all' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-09-29_14-06-54a' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-09-29_17-10-17b' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-09-29_15-38-03c' 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-09-29_11-24-04f' 
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-09-29_12-51-08all' 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-09-29_14-04-11a' 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-09-29_17-11-09b' 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-09-29_15-38-25c' 
