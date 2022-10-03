#!/bin/sh
python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc none  -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm self_supervised
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm self_supervised
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm self_supervised
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm self_supervised

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm self_supervised 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm self_supervised 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm self_supervised

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm self_supervised 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 3 -pos_enc none  -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 3 -pos_enc none -tstcc_tm self_supervised
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 3 -pos_enc none -tstcc_tm self_supervised
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 3 -pos_enc none -tstcc_tm self_supervised
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 3 -pos_enc none -tstcc_tm self_supervised

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm self_supervised 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm self_supervised 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm self_supervised

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm self_supervised 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc none  -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm self_supervised
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm self_supervised
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm self_supervised
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm self_supervised

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm self_supervised 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm self_supervised 

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm self_supervised 
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm self_supervised

python3 main.py -c configs/config_c_f.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_all.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_a.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_b.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_c_c.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm self_supervised 

# CHANGE THE NEPOCHS TO 40 for TSTCC