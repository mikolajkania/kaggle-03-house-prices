import os
import sys
import yaml

sys.path.extend(os.pardir)

from src.data.load import DataLoader

# PARAMS

params_path = sys.argv[1]
params = yaml.safe_load(open(params_path))['prepare']['data']

data = DataLoader(params['path'])
all_df = data.load()
print(all_df.head())
