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
    h.update(json.dumps(params).encode('utf-8'))
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
    
    end_year = int(year_month_range[1][:4])
    end_month = int(year_month_range[1][4:6])
    end_date = datetime.datetime(year=end_year, 
                                 month=end_month, 
                                 day=monthrange(end_year, end_month)[1])
    
    return [item.date() for item in pd.date_range(start=start_date, end=end_date)]
