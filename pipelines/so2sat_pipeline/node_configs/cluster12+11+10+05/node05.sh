TF_CONFIG=$(cat node_configs/cluster12+11+10+05/node05_tfconfig.json)
export TF_CONFIG
UMLAUT_CONFIG=$(cat node_configs/cluster12+11+10+05/node05_umlautconfig.json)
export UMLAUT_CONFIG
python3 main.py