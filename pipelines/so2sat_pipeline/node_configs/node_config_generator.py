import os


def generate_config(nodes):
    foldername=f"cluster{'+'.join(nodes)}"
    folderpath = os.path.join(os.getcwd(), foldername)

    os.mkdir(path=folderpath)

    generate_umlautconfig('main', nodes[0], 0, folderpath)
    generate_shellscript(nodes[0], foldername, folderpath)
    generate_tfconfig(nodes[0], 0, nodes, folderpath)

    for i in range(1, len(nodes)):
        generate_umlautconfig('main', nodes[i], i, folderpath)
        generate_shellscript(nodes[i], foldername, folderpath)
        generate_tfconfig(nodes[i], i, nodes, folderpath)


def generate_umlautconfig(role, node, worker_number, folderpath):
    umlautconfig = f"""{{
  "role": "{role}",
  "main_address": "node-{node}.delab.i.hpi.de",
  "main_port": 35353,
  "worker_number": {worker_number}
}}"""
    with open(os.path.join(folderpath, f"node{node}_umlautconfig.json"), 'w') as f:
        f.write(umlautconfig)


def generate_shellscript(node, foldername, folderpath):
    shellscript = f"""TF_CONFIG=$(cat node_configs/{foldername}/node{node}_tfconfig.json)
export TF_CONFIG
UMLAUT_CONFIG=$(cat node_configs/{foldername}/node{node}_umlautconfig.json)
export UMLAUT_CONFIG
python3 main.py"""
    with open(os.path.join(folderpath, f"node{node}.sh"), 'w') as f:
        f.write(shellscript)


def generate_tfconfig(node, index, nodes, folderpath):
    tfconfig = f"""{{
  "cluster": {{
    "worker": {str([f"node-{node}.delab.i.hpi.de:25252" for node in nodes])}
  }},
  "task": {{
    "type": "worker",
    "index": {index}
  }}
}}"""
    with open(os.path.join(folderpath, f"node{node}_tfconfig.json"), 'w') as f:
        f.write(tfconfig)


if __name__ == "__main__":
    nodes = ["10"]
    generate_config(nodes)
