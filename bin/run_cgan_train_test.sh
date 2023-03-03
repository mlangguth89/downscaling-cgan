#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-80:00:00
#SBATCH --mem=50gb
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cgan-test
#SBATCH --partition cnu
#SBATCH --output=logs/slurm-%A.out

# GPU can handle 200 x 200 x 64 x 2 arrays with 100gb

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
echo "running model"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

# either run the script to train your model
srun python -m dsrnngan.test
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
 
