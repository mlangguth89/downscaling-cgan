#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-0:00:00
#SBATCH --mem=150gb
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=qmap
#SBATCH --partition short,dmm,compute
#SBATCH --output=logs/qmap-%A_%a.out
#SBATCH --array=1,15

source ~/.bashrc
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

# print out stuff to tell you when the script is running
echo running quantile mapping
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

# either run the script to train your model
# srun python -m scripts.quantile_mapping --num-lat-lon-chunks ${SLURM_ARRAY_TASK_ID} --log-folder /user/work/uz22147/logs/cgan/7c4126e641f81ae0_medium-cl100-final-nologs --model-number 217600 --output-folder plots/quantile_map_plots --plot;
srun python -m scripts.quantile_mapping --num-lat-lon-chunks ${SLURM_ARRAY_TASK_ID}  --log-folder /user/work/uz22147/logs/cgan/7c4126e641f81ae0_medium-cl100-final-nologs --model-number 217600 --output-folder plots/quantile_map_plots --save-qmapper;

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
