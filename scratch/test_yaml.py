import yaml
import numpy as np
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
delta = config['dp']['delta']
print(f"Type of delta: {type(delta)}")
print(f"Value of delta: {delta}")
try:
    print(f"Log of delta: {np.log(delta)}")
except Exception as e:
    print(f"Error: {e}")
