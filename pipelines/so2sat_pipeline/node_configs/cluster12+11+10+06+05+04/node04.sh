TF_CONFIG=$(cat node_configs/cluster12+11+10+06+05+04/node04_tfconfig.json)
export TF_CONFIG
UMLAUT_CONFIG=$(cat node_configs/cluster12+11+10+06+05+04/node04_umlautconfig.json)
export UMLAUT_CONFIG
python3 main.py