TF_CONFIG=$(cat node_configs/cluster10+09+08/node10_tfconfig.json)
export TF_CONFIG
UMLAUT_CONFIG=$(cat node_configs/cluster10+09+08/node10_umlautconfig.json)
export UMLAUT_CONFIG
python3 main.py