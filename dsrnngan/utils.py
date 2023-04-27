import os
import sys
from pathlib import Path
import hashlib
import json
import yaml
from glob import glob
import datetime
import pandas as pd
import numpy as np
from calendar import monthrange


def hash_dict(params: dict):
    h = hashlib.shake_256()
    h.update(json.dumps(params, sort_keys=True).encode('utf-8'))
    return h.hexdigest(8)


def load_yaml_file(fpath: str):  
    with open(fpath, 'r') as f:
        data = yaml.safe_load(f)
        
    return data

def write_to_yaml(fpath: str, data: dict):
    
    with open(fpath, 'w+') as ofh:
        yaml.dump(data, ofh, default_flow_style=False)
        
def date_range_from_year_month_range(year_month_range):
    
    start_date = datetime.datetime(year=int(year_month_range[0][:4]), month=int(year_month_range[0][4:6]), day=1)
    
    end_year = int(year_month_range[-1][:4])
    end_month = int(year_month_range[-1][4:6])
    end_date = datetime.datetime(year=end_year, 
                                 month=end_month, 
                                 day=monthrange(end_year, end_month)[1])
    
    return [item.date() for item in pd.date_range(start=start_date, end=end_date)]

def get_best_model_number(log_folder: str, metric_column_name: str='CRPS_no_pooling'):
    """
    Get model number that has lowest value according to metric specified (defaults to CRPS)

    Args:
        log_folder (str): Path to evaluation results
        metric_column_name (str): Name of column to optimise on. Defaults to CRPS_no_pooling 

    Returns:
        int: Model number
    """
    df = pd.read_csv(os.path.join(log_folder, 'eval_validation.csv'))

    # find model_number with lowest CRPS
    min_crps = df['CRPS_no_pooling'].min()
    model_number = df[df.CRPS_no_pooling == min_crps]['N'].values[0]
    
    return model_number
