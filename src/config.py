import os, yaml

# assume your project root is cwd, and config lives in ./config/config.yaml
CONFIG_PATH = os.path.join(os.getcwd(), "config", "config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)