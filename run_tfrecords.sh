#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-20:00:00
#SBATCH --mem=50gb
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=tfrecords-creation
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
srun python -m dsrnngan.tfrecords_generator --fcst-hours 18 --records-folder /user/work/uz22147/tfrecords

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
 
