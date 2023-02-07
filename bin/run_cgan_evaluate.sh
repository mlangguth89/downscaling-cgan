#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=50gb
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cgan-eval
#SBATCH --partition cnu
#SBATCH --output=logs/slurm-%A_%a.out

source ~/.bashrc
source ~/.initConda.sh
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST

# checkout what the gpus are doing
nvidia-smi --query-gpu=gpu_name,driver_version,memory.free,memory.total --format=csv

nvidia-smi
module load lang/cuda/11.2-cudnn-8.1
nvidia-smi

# print out stuff to tell you when the script is running
echo running model
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

# either run the script to train your model
srun python -m dsrnngan.main --evaluate --model-numbers 153600 147200 140800 134400 128000 121600 115200 108800 --no-train --records-folder /user/work/uz22147/tfrecords/d34d309eb0e00b04 --num-images 1000 --ensemble-size 50 --val-ym-start 201806 --val-ym-end 201905
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
 
