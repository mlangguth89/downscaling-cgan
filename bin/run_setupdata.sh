#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=20gb
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cgan-setupdata
#SBATCH --partition compute
#SBATCH --output=logs/slurm-%A.out

source ~/.bashrc
source ~/.initConda.sh
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

# print out stuff to tell you when the script is running
echo running setupdata
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

# either run the script to train your model
srun python -m dsrnngan.setupdata

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
 
