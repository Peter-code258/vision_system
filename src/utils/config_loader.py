# src/utils/config_loader.py
import yaml, os

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_default(config_dir="configs"):
    path = os.path.join(config_dir, "default.yaml")
    return load_yaml(path) if os.path.exists(path) else {}
