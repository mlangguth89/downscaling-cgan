#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=20gb
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cgan-data
#SBATCH --partition compute
#SBATCH --array=2016
#SBATCH --output=logs/slurm-%A_%a.out

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
srun python -m dsrnngan.data.data --years ${SLURM_ARRAY_TASK_ID} --months 1 2 3 4 5 6 7 8 9 10 11 12  --obs-data-source imerg --output-dir /bp1/geog-tropical/users/uz22147/east_africa_data --min-latitude -12 --max-latitude 17 --min-longitude 22 --max-longitude 52

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
 
