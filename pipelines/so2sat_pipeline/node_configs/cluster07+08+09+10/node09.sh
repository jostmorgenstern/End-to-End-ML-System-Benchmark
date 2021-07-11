TF_CONFIG=$(cat node_configs/cluster07+08+09+10/node09_tfconfig.json)
export TF_CONFIG
UMLAUT_CONFIG=$(cat node_configs/cluster07+08+09+10/node09_umlautconfig.json)
export UMLAUT_CONFIG
python3 main.py