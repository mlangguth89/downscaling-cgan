#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-16:00:00
#SBATCH --mem=10gb
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=ifs-unpack
#SBATCH --partition short

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

# print out stuff to tell you when the script is running
echo running ifs unpacking
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

# either run the script to train your model
srun python -m scripts.unpack_ifs_data 

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
 
