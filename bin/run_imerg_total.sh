#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-10:00:00
#SBATCH --mem=10gb
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=monthly_rainfall
#SBATCH --partition short
#SBATCH --array=2009
#SBATCH --output=logs/imerg-total-%A_%a.out

source ~/.bashrc
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

# print out stuff to tell you when the script is running
echo running imerg monthly rainfall
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

# either run the script to train your model
srun python -m scripts.imerg_monthly_rainfall --year ${SLURM_ARRAY_TASK_ID}

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
 
