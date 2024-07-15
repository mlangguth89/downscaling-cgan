#!/bin/bash

  ml --force purge
  ml use $OTHERSTAGES
  ml Stages/2022

  ml GCCcore/.11.2.0
  ml GCC/11.2.0
  ml ParaStationMPI/5.5.0-1
  ml tqdm/4.62.3
  ml git/2.33.1-nodocs
  ml Jupyter/2022.3.3
  # ml ESMF/8.2.0    # may require version 8.4.1 which is only available for Stages/2023
  ml netcdf4-python/1.5.7-serial
  ml h5py/3.5.0-serial
  ml scikit-image/0.18.3
  ml scikit-learn/1.0.1
  ml SciPy-bundle/2021.10
  ml xarray/0.20.1
  ml dask/2021.9.1
  ml TensorFlow/2.6.0-CUDA-11.5
  ml Cartopy/0.20.0
  ml Graphviz/2.49.3

