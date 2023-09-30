#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=100gb
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cgan-setupdata
#SBATCH --partition compute,dmm,short
#SBATCH --output=logs/setupdata-%A_%a.out
#SBATCH --array=1-10

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
srun python -m dsrnngan.data.setupdata --output-dir /user/work/uz22147/quantile_training_data --model-folder /user/work/uz22147/logs/cgan/7c4126e641f81ae0_medium-cl100-final-nologs --ym-index ${SLURM_ARRAY_TASK_ID}

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
 
