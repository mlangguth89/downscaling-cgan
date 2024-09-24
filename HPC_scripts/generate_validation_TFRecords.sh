#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=generate_validation_TFRecords-out.%j
#SBATCH --error=generate_validation_TFRecords-err.%j
##SBATCH --time=02:00:00
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
##SBATCH --partition=batch
##SBATCH --partition=gpus
##SBATCH --partition=develgpus
#SBATCH --partition=booster
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
VIRT_ENV_NAME=venv_JB

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

records_folder=/p/scratch/atmo-rep/data/downscaling/downscaling_tfrecords/validation_data/
data_config=${BASE_DIR}/config/data_config.yaml
model_config=${BASE_DIR}/config/model_config.yaml



# run job
srun --overlap python -m dsrnngan.data.tfrecords_generator --records-folder ${records_folder} --data-config-path ${data_config} --model-config-path ${model_config} --val

