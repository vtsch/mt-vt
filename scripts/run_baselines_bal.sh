#!/bin/sh
python3 main.py -c configs/config_c_f_bal.json -exp simple_ae -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ae -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ae -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ae -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ae -n_clusters 2 -pos_enc none 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ae -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ae -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ae -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ae -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ae -n_clusters 2 -pos_enc delta_days 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ae -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ae -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ae -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ae -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ae -n_clusters 2 -pos_enc absolute_days 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ae -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ae -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ae -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ae -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ae -n_clusters 2 -pos_enc age_pos_enc 

python3 main.py -c configs/config_c_f_bal.json -exp simple_ae -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp simple_ae -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp simple_ae -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp simple_ae -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp simple_ae -n_clusters 2 -pos_enc learnable_pos_enc 

python3 main.py -c configs/config_c_f_bal.json -exp cnn -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_all_bal.json -exp cnn -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_a_bal.json -exp cnn -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_b_bal.json -exp cnn -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_c_bal.json -exp cnn -n_clusters 2 -pos_enc none 

python3 main.py -c configs/config_c_f_bal.json -exp cnn -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_all_bal.json -exp cnn -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_a_bal.json -exp cnn -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_b_bal.json -exp cnn -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_c_bal.json -exp cnn -n_clusters 2 -pos_enc delta_days 

python3 main.py -c configs/config_c_f_bal.json -exp cnn -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_all_bal.json -exp cnn -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_a_bal.json -exp cnn -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_b_bal.json -exp cnn -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_c_bal.json -exp cnn -n_clusters 2 -pos_enc absolute_days 

python3 main.py -c configs/config_c_f_bal.json -exp cnn -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp cnn -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp cnn -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp cnn -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp cnn -n_clusters 2 -pos_enc age_pos_enc 

python3 main.py -c configs/config_c_f_bal.json -exp cnn -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp cnn -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp cnn -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp cnn -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp cnn -n_clusters 2 -pos_enc learnable_pos_enc 

python3 main.py -c configs/config_c_f_bal.json -exp lstm -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_all_bal.json -exp lstm -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_a_bal.json -exp lstm -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_b_bal.json -exp lstm -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_c_bal.json -exp lstm -n_clusters 2 -pos_enc none 

python3 main.py -c configs/config_c_f_bal.json -exp lstm -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_all_bal.json -exp lstm -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_a_bal.json -exp lstm -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_b_bal.json -exp lstm -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_c_bal.json -exp lstm -n_clusters 2 -pos_enc absolute_days

python3 main.py -c configs/config_c_f_bal.json -exp lstm -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp lstm -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp lstm -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp lstm -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp lstm -n_clusters 2 -pos_enc age_pos_enc 

python3 main.py -c configs/config_c_f_bal.json -exp lstm -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_all_bal.json -exp lstm -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_a_bal.json -exp lstm -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_b_bal.json -exp lstm -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_c_bal.json -exp lstm -n_clusters 2 -pos_enc delta_days 

python3 main.py -c configs/config_c_f_bal.json -exp lstm -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp lstm -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp lstm -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp lstm -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp lstm -n_clusters 2 -pos_enc learnable_pos_enc 

python3 main.py -c configs/config_c_f_bal.json -exp simple_transformer -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_all_bal.json -exp simple_transformer -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_a_bal.json -exp simple_transformer -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_b_bal.json -exp simple_transformer -n_clusters 2 -pos_enc none 
python3 main.py -c configs/config_c_c_bal.json -exp simple_transformer -n_clusters 2 -pos_enc none 

python3 main.py -c configs/config_c_f_bal.json -exp simple_transformer -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_all_bal.json -exp simple_transformer -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_a_bal.json -exp simple_transformer -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_b_bal.json -exp simple_transformer -n_clusters 2 -pos_enc absolute_days 
python3 main.py -c configs/config_c_c_bal.json -exp simple_transformer -n_clusters 2 -pos_enc absolute_days 

python3 main.py -c configs/config_c_f_bal.json -exp simple_transformer -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp simple_transformer -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp simple_transformer -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp simple_transformer -n_clusters 2 -pos_enc age_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp simple_transformer -n_clusters 2 -pos_enc age_pos_enc 

python3 main.py -c configs/config_c_f_bal.json -exp simple_transformer -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_all_bal.json -exp simple_transformer -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_a_bal.json -exp simple_transformer -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_b_bal.json -exp simple_transformer -n_clusters 2 -pos_enc delta_days 
python3 main.py -c configs/config_c_c_bal.json -exp simple_transformer -n_clusters 2 -pos_enc delta_days 

python3 main.py -c configs/config_c_f_bal.json -exp simple_transformer -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_all_bal.json -exp simple_transformer -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_a_bal.json -exp simple_transformer -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_b_bal.json -exp simple_transformer -n_clusters 2 -pos_enc learnable_pos_enc 
python3 main.py -c configs/config_c_c_bal.json -exp simple_transformer -n_clusters 2 -pos_enc learnable_pos_enc 
