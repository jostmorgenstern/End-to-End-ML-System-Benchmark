TF_CONFIG=$(cat node_configs/cluster1+2+3/node1_tfconfig.json)
export TF_CONFIG
UMLAUT_CONFIG=$(cat node_configs/cluster1+2+3/node1_umlautconfig.json)
export UMLAUT_CONFIG
python3 main.py