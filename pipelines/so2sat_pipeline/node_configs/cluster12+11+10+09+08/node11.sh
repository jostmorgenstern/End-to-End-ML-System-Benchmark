TF_CONFIG=$(cat node_configs/cluster12+11+10+09+08/node11_tfconfig.json)
export TF_CONFIG
UMLAUT_CONFIG=$(cat node_configs/cluster12+11+10+09+08/node11_umlautconfig.json)
export UMLAUT_CONFIG
python3 main.py