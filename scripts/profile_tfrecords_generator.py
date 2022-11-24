import os, sys
import tempfile
import cProfile, pstats, io
from pstats import SortKey
from pathlib import Path
import numpy as np

HOME = Path(os.getcwd()).parents[0]

sys.path.insert(1, str(HOME))

from dsrnngan.tfrecords_generator import write_data

if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_paths = {'GENERAL': {'IMERG': '/bp1/geog-tropical/users/uz22147/east_africa_data/IMERG/half_hourly/final/',
                                            'ERA5': '/bp1/geog-tropical/data/ERA-5/day',
                                            'IFS': '/bp1/geog-tropical/users/uz22147/east_africa_data/IFS',
                                            'OROGRAPHY': '/bp1/geog-tropical/users/uz22147/east_africa_data/constants/h_HRES_EAfrica.nc',
                                            'LSM': '/bp1/geog-tropical/users/uz22147/east_africa_data/constants/lsm_HRES_EAfrica.nc',
                                            'CONSTANTS': '/bp1/geog-tropical/users/uz22147/east_africa_data/constants'},
                            'TFRecords': {'tfrecords_path': tmpdirname}}
        
        write_data(2017, forecast_data_source='ifs', observational_data_source='imerg', hours=[18], img_chunk_width=10, img_size=10,
                num_class=4,
                log_precip=True,
                fcst_norm=False,
                scaling_factor=1,
                latitude_range=np.arange(0.05, 1.05, 0.1),
                longitude_range=np.arange(33.05, 34.05, 0.1), debug=True)
        
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(0.01) #Just the top 1% of stats
        print(s.getvalue())