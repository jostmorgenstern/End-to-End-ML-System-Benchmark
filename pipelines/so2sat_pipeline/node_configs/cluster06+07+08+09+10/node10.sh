TF_CONFIG=$(cat node_configs/cluster06+07+08+09+10/node10_tfconfig.json)
export TF_CONFIG
UMLAUT_CONFIG=$(cat node_configs/cluster06+07+08+09+10/node10_umlautconfig.json)
export UMLAUT_CONFIG
python3 main.py