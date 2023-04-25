#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=100gb
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cgan-eval
#SBATCH --partition cnu
#SBATCH --output=logs/slurm-%A.out

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
srun python -m dsrnngan.main --no-train --eval-model-numbers 275200 --save-generated-samples --records-folder /user/work/uz22147/tfrecords/f6998afe16c9f955 --num-images 4000 --ensemble-size 10 --val-ym-start 201806 --val-ym-end 201905
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

