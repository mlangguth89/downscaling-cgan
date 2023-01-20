#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-5:00:00
#SBATCH --mem=20gb
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cgan-data
#SBATCH --partition compute
#SBATCH --array=2000-2020

source ~/.bashrc
source ~/.initConda.sh
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

# print out stuff to tell you when the script is running
echo running data collection
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

# either run the script to train your model
srun python -m dsrnngan.data --years ${SLURM_ARRAY_TASK_ID} --months 1 2 3 4 5 6 7 8  --obs-data-source imerg --output-dir /bp1/geog-tropical/users/uz22147/east_africa_data --min-latitude -12 --max-latitude 16 --min-longitude 22 --max-longitude 50

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
 
