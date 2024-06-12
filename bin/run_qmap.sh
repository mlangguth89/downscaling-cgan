#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-0:00:00
#SBATCH --mem=150gb
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=qmap
#SBATCH --partition shared
#SBATCH --output=logs/qmap-%A_%a.out

source ~/.bashrc
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`

# print out stuff to tell you when the script is running
echo running quantile mapping
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

# either run the script to train your model
# srun python -m scripts.quantile_mapping --num-lat-lon-chunks ${SLURM_ARRAY_TASK_ID} --log-folder /user/work/uz22147/logs/cgan/7c4126e641f81ae0_medium-cl100-final-nologs --model-number 217600 --output-folder plots/quantile_map_plots --plot;
srun python -m scripts.quantile_mapping --num-lat-lon-chunks 2 --model-eval-folder /network/group/aopp/predict/HMC005_ANTONIO_EERIE/cgan_data/7c4126e641f81ae0_medium-cl100-final-nologs/n8640_202010-202109_45682_e1 --model-number 217600 --save-data;
srun python -m scripts.quantile_mapping --num-lat-lon-chunks 3 --model-eval-folder /network/group/aopp/predict/HMC005_ANTONIO_EERIE/cgan_data/7c4126e641f81ae0_medium-cl100-final-nologs/n8640_202010-202109_45682_e1 --model-number 217600 --save-data;

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
