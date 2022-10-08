#!/bin/sh
python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 2 -pos_enc none 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 2 -pos_enc absolute_days 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 2 -pos_enc age_pos_enc 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 2 -pos_enc delta_days 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 2 -pos_enc learnable_pos_enc 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 3 -pos_enc none 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 3 -pos_enc none 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 3 -pos_enc none 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 3 -pos_enc none 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 3 -pos_enc none 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 3 -pos_enc absolute_days 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 3 -pos_enc absolute_days 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 3 -pos_enc absolute_days 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 3 -pos_enc absolute_days 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 3 -pos_enc absolute_days 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 3 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 3 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 3 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 3 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 3 -pos_enc age_pos_enc 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 3 -pos_enc delta_days 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 3 -pos_enc delta_days 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 3 -pos_enc delta_days 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 3 -pos_enc delta_days 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 3 -pos_enc delta_days 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 3 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 3 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 3 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 3 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 3 -pos_enc learnable_pos_enc 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 4 -pos_enc none 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 4 -pos_enc none 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 4 -pos_enc none 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 4 -pos_enc none 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 4 -pos_enc none 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 4 -pos_enc delta_days 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 4 -pos_enc delta_days 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 4 -pos_enc delta_days 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 4 -pos_enc delta_days 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 4 -pos_enc delta_days 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 4 -pos_enc absolute_days 
python3 main.py -c configs/config_c_all_bal.jso_baln -exp simple_ac -n_clusters 4 -pos_enc absolute_days 
python3 main.py -c configs/config_c_a_bal.json -e_balxp simple_ac -n_clusters 4 -pos_enc absolute_days 
python3 main.py -c configs/config_c_b_bal.json -exp s_balimple_ac -n_clusters 4 -pos_enc absolute_days 
python3 main.py -c configs/config_c_c_bal.json -exp simpl_bale_ac -n_clusters 4 -pos_enc absolute_days 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 4 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 4 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 4 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 4 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 4 -pos_enc age_pos_enc 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ac -n_clusters 4 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ac -n_clusters 4 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ac -n_clusters 4 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ac -n_clusters 4 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ac -n_clusters 4 -pos_enc learnable_pos_enc 