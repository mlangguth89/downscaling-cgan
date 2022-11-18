"""
Script to unpack tar files downloaded from ECMWF
"""

import shutil
import tarfile
import tempfile
import os
import re
from glob import glob
from tqdm import tqdm

ifs_output_dir = '/bp1/geog-tropical/users/uz22147/east_africa_data/IFS'

if __name__ == '__main__':
    
    tar_fps = glob('/user/work/ua20683/bobby/*.tar')
    
    for fp in tqdm(tar_fps, total=len(tar_fps)):
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                tar = tarfile.open(fp, "r")
                tar.extractall(tmp_dir)
                
                data_fps = glob(os.path.join(tmp_dir, '*.nc'))

                for data_fp in data_fps:
                    
                    filename = data_fp.split('/')[-1]
                    var = filename.split('_')[0]

                    top_level_folder = re.sub(r'([0-9]*[a-z]+)[0-9]*', r'\1', var)
                    subdirectory = True
                            
                    if top_level_folder == var:
                        subdirectory = False
                    
                    top_level_folder = os.path.join(ifs_output_dir, top_level_folder)
                    
                    if not os.path.isdir(top_level_folder):
                        os.mkdir(top_level_folder)
                            
                    if subdirectory:
                        subdir = os.path.join(top_level_folder, var)
                        
                        if not os.path.isdir(subdir):
                            os.mkdir(subdir)
                            
                        shutil.copyfile(data_fp, os.path.join(subdir, filename))

                    else:
                        shutil.copyfile(data_fp, os.path.join(top_level_folder, filename))
            except PermissionError as e:
                print(f'Unable to open {fp}')

