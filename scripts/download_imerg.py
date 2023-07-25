import subprocess
import os
import tempfile
from tqdm import tqdm
from dsrnngan.data.data import get_imerg_filepaths, load_hdf5_file, filter_by_lat_lon
from calendar import monthrange

from argparse import ArgumentParser


parser = ArgumentParser(description='Script for quantile mapping.')

parser.add_argument('--year', type=int)
parser.add_argument('--month', type=int)
parser.add_argument('--output-folder', type=str, help='Folder to save data in', default=None)

latitude_vals = [-14.05, 16.05]
longitude_vals = [20,55]
    
args = parser.parse_args()

year = args.year
month=args.month

for day in tqdm(range(1, monthrange(year, month)[1]+1)):
    with tempfile.TemporaryDirectory() as tmpdirname:
        for hour in range(24):

            fps = get_imerg_filepaths(year=year, month=month, day=day, 
                                    hour=hour, imerg_data_dir=f"https://arthurhouhttps.pps.eosdis.nasa.gov/text/gpmdata/{year}/{month:02d}/{day:02d}/imerg/", 
                                    file_ending='.HDF5')
            fp = fps[0]
            output_fp = get_imerg_filepaths(year=year, month=month, day=day,
                                    hour=hour, imerg_data_dir=tmpdirname, 
                                    file_ending='.HDF5')[0]

            sp_args = ["curl", fp, "--user", "bobbyantonio@gmail.com:bobbyantonio@gmail.com", "--output", output_fp]
            process = subprocess.Popen(sp_args,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            process.wait()
            
            # # Convert to nc
            ds =load_hdf5_file(output_fp)
            ds = filter_by_lat_lon(ds, lon_range=longitude_vals, 
                                    lat_range=latitude_vals)
            path_suffix = output_fp.split('/')[-1]
            imerg_path = os.path.join(args.output_folder, path_suffix).replace('.HDF5', '.nc')
            ds.to_netcdf(imerg_path)
        


