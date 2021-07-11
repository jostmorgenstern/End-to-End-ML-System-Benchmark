TF_CONFIG=$(cat node_configs/cluster07+08+09+10/node07_tfconfig.json)
export TF_CONFIG
UMLAUT_CONFIG=$(cat node_configs/cluster07+08+09+10/node07_umlautconfig.json)
export UMLAUT_CONFIG
python3 main.py