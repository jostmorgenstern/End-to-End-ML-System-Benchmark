TF_CONFIG=$(cat node_configs/cluster05+06+07+08+09+10/node06_tfconfig.json)
export TF_CONFIG
UMLAUT_CONFIG=$(cat node_configs/cluster05+06+07+08+09+10/node06_umlautconfig.json)
export UMLAUT_CONFIG
python3 main.py