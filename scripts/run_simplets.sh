#!/bin/sh
python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 2 -pos_enc none 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 2 -pos_enc absolute_days 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 2 -pos_enc age_pos_enc 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 2 -pos_enc delta_days 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 2 -pos_enc learnable_pos_enc 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 3 -pos_enc none 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 3 -pos_enc none 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 3 -pos_enc none 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 3 -pos_enc none 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 3 -pos_enc none 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 3 -pos_enc absolute_days 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 3 -pos_enc absolute_days 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 3 -pos_enc absolute_days 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 3 -pos_enc absolute_days 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 3 -pos_enc absolute_days 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 3 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 3 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 3 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 3 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 3 -pos_enc age_pos_enc 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 3 -pos_enc delta_days 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 3 -pos_enc delta_days 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 3 -pos_enc delta_days 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 3 -pos_enc delta_days 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 3 -pos_enc delta_days 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 3 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 3 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 3 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 3 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 3 -pos_enc learnable_pos_enc 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 4 -pos_enc none 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 4 -pos_enc none 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 4 -pos_enc none 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 4 -pos_enc none 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 4 -pos_enc none 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 4 -pos_enc delta_days 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 4 -pos_enc delta_days 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 4 -pos_enc delta_days 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 4 -pos_enc delta_days 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 4 -pos_enc delta_days 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 4 -pos_enc absolute_days 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 4 -pos_enc absolute_days 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 4 -pos_enc absolute_days 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 4 -pos_enc absolute_days 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 4 -pos_enc absolute_days 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 4 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 4 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 4 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 4 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 4 -pos_enc age_pos_enc 

python3 main.py -c configs/config_c_f.json -exp simple_transformer -n_clusters 4 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_all.json -exp simple_transformer -n_clusters 4 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_a.json -exp simple_transformer -n_clusters 4 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_b.json -exp simple_transformer -n_clusters 4 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_c.json -exp simple_transformer -n_clusters 4 -pos_enc learnable_pos_enc 