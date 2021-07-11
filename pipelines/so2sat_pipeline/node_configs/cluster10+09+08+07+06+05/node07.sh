TF_CONFIG=$(cat node_configs/cluster10+09+08+07+06+05/node07_tfconfig.json)
export TF_CONFIG
UMLAUT_CONFIG=$(cat node_configs/cluster10+09+08+07+06+05/node07_umlautconfig.json)
export UMLAUT_CONFIG
python3 main.py