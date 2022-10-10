#!/bin/sh
python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 2 -pos_enc none  -tstcc_tm train_linear -tstcc_dir '22-10-07_15-35-19' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-10-07_15-36-20' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-10-07_15-37-18' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-10-07_15-38-17' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-10-07_15-39-18' 

python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_15-40-17' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_15-41-16' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_15-42-13' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_15-43-10' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_15-44-07' 

python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_15-45-04' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_15-46-00' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_15-46-57' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_15-47-54' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_15-48-51' 

python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_15-49-46' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_15-50-44' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_15-51-39' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_15-52-32' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_15-53-26' 

python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_15-54-22' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_15-55-19' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_15-56-14' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_15-57-11' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_15-58-08' 


python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 3 -pos_enc none  -tstcc_tm train_linear -tstcc_dir '22-10-07_15-59-06' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 3 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-10-07_16-00-01' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 3 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-10-07_16-00-59' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 3 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-10-07_16-01-55' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 3 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-10-07_16-02-49' 

python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-03-46' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-04-44' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-05-39' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-06-39' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 3 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-07-37' 

python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-08-28' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-09-28' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-10-22' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-11-15' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 3 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-12-07' 

python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-13-01' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-13-53' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-14-46' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-15-39' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 3 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-16-31' 

python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-17-23' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-18-22' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-19-23' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-20-26'
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 3 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-21-27' 
	

python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 4 -pos_enc none  -tstcc_tm train_linear -tstcc_dir '22-10-07_16-22-26' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-10-07_16-23-19' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-10-07_16-24-15' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-10-07_16-25-12' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm train_linear -tstcc_dir '22-10-07_16-26-08' 

python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-27-05' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-28-01' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-29-00' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-30-27' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-31-28' 

python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-32-24' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-33-21' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-34-31' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-35-39' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm train_linear -tstcc_dir '22-10-07_16-36-58' 

python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-38-01' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-39-06' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-40-09' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-41-13' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-42-18' 

python3 main.py -c configs/config_c_f_bal.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-43-23' 
python3 main.py -c configs/config_c_all_bal.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-44-27' 
python3 main.py -c configs/config_c_a_bal.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-45-30' 
python3 main.py -c configs/config_c_b_bal.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-46-35' 
python3 main.py -c configs/config_c_c_bal.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm train_linear -tstcc_dir '22-10-07_16-47-33' 
