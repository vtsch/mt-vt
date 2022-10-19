#!/bin/sh

python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc none -tstcc_tm self_supervised
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc absolute_days -tstcc_tm self_supervised
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc delta_days -tstcc_tm self_supervised
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc age_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 2 -pos_enc learnable_pos_enc -tstcc_tm self_supervised

python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 4 -pos_enc none -tstcc_tm self_supervised
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 4 -pos_enc absolute_days -tstcc_tm self_supervised
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 4 -pos_enc delta_days -tstcc_tm self_supervised
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 4 -pos_enc age_pos_enc -tstcc_tm self_supervised
python3 main.py -c configs/config_furst.json -exp ts_tcc -n_clusters 4 -pos_enc learnable_pos_enc -tstcc_tm self_supervised
