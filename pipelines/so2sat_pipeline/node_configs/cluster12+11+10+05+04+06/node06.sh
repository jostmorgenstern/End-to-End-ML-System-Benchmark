TF_CONFIG=$(cat node_configs/cluster12+11+10+05+04+06/node06_tfconfig.json)
export TF_CONFIG
UMLAUT_CONFIG=$(cat node_configs/cluster12+11+10+05+04+06/node06_umlautconfig.json)
export UMLAUT_CONFIG
python3 main.py