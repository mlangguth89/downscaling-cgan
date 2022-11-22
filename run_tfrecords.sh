#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-7:00:00
#SBATCH --mem=50gb
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=2018-2021-tfrecords
#SBATCH --partition short

source ~/.bashrc
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

# print out stuff to tell you when the script is running
echo running model
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

# either run the script to train your model
# srun python -m dsrnngan.tfrecords_generator --fcst-data-source era5 --obs-data-source imerg --startdate 20180101 --enddate 20211231 --fcst-hours 18 --log-precip --fcst-norm --img-size 200 --img-chunk-width 200 --scaling-factor 1 
srun python -m dsrnngan.tfrecords_generator --fcst-data-source ifs --obs-data-source imerg --train-start-date 20160301 --train-end-date 20170228 --test-start-date 20170301 --test-end-date 20180228 --log-precip --fcst-norm --img-size 200 --img-chunk-width 200 --scaling-factor 1 --records-folder /user/work/uz22147/tfrecords


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
 
