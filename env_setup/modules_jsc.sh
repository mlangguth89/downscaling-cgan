#!/bin/bash

  ml --force purge
  ml use $OTHERSTAGES
  ml Stages/2023

  ml GCCcore/.11.3.0
  ml GCC/11.3.0
  #ml Intel/2022.1.0
  ml ParaStationMPI/5.8.1-1
  ml tqdm/4.64.0
  ml git/2.36.0-nodocs
  # ml Jupyter/2022.3.3
  #ml ESMF/8.2.0    # may require version 8.4.1 which is only available for Stages/2023
  ml CDO/2.1.1
  ml netcdf4-python/1.6.1-serial
  ml h5py/3.7.0-serial
  ml scikit-image/0.19.3
  ml scikit-learn/1.1.2
  ml SciPy-Stack/2022a
  ml dask/2022.12.0
  ml TensorFlow/2.11.0-CUDA-11.7
  ml Cartopy/0.21.0
  ml Graphviz/5.0.0

