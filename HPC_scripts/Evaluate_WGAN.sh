#!/bin/bash -x
#SBATCH --account=hclimrep
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/Evaluate_WGAN.%j-out
#SBATCH --error=logs/Evaluate_WGAN.%j-err
##SBATCH --time=02:00:00
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
##SBATCH --partition=batch
#SBATCH --partition=gpus
##SBATCH --partition=develgpus
##SBATCH --partition=booster
##SBATCH --partition=develbooster
##SBATCH --mail-type=ALL
##SBATCH --mail-user=e.pavel@fz-juelich.de

# environmental variables to support cpus_per_task with Slurm>22.05
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

# basic directories
WORK_DIR=$(pwd)
BASE_DIR=$(dirname "${WORK_DIR}")

# Name of virtual environment
VENV_DIR=${BASE_DIR}/virtual_envs/
VIRT_ENV_NAME=venv_JC

# Loading mouldes
source ../env_setup/modules_jsc.sh
# Activate virtual environment if needed (and possible)
if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ${VENV_DIR}/${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ${VENV_DIR}/${VIRT_ENV_NAME}/bin/activate
   else
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi

records_folder=/p/scratch/atmo-rep/data/downscaling/downscaling_tfrecords/evaluation_data/33b737e4a85a41aa
#/p/scratch/atmo-rep/data/downscaling/downscaling_tfrecords/validation_data/33b737e4a85a41aa
data_config=${BASE_DIR}/config/data_config.yaml
model_config=${BASE_DIR}/config/model_config.yaml
#model_folder=/p/scratch/hclimrep/pavel1/0aad51a8f3848213_downscaling_harris_wgan_bs16
model_folder=/p/scratch/hclimrep/pavel1/0aad51a8f3848213_downscaling_harris_wgan_bs16_alt
# --eval-months-idx 1,2,4-6,9

# run job
srun --overlap python -m dsrnngan.main --records-folder ${records_folder} --model-folder ${model_folder} --eval-ensemble-size 100  --no-train --eval-model-idx 2 --eval-blitz --skip-interval 1 --eval-months-idx 9-11 --lon-min -19.0833333333333 --lon-max 35.0833333333333  --lat-min 24.9166666666667  --lat-max 73.8333333333333


