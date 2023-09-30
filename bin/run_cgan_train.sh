#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=8-00:00:00
#SBATCH --mem=300gb
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cgan-train
#SBATCH --partition cnu
#SBATCH --output=logs/cgan-train-%A.out


# GPU can handle 200 x 200 x 64 x 2 arrays with 100gb
# Memory needs to be around 400gb to get decent performance, particularly it seems when passing in larger datasets
# --gpus-per-task=1 --gres=gpu:A100:1
# try with --gres=gpu:A100:1

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
# srun python -m dsrnngan.main --num-samples 320000 --eval-short --restart --records-folder /user/work/uz22147/tfrecords/6baa8b10ec61eaed --model-config-path /user/home/uz22147/repos/downscaling-cgan/config/model_config_medium-cl500.yaml --output-suffix medium-cl500-sqrt-norm-2 --num-images 2900 --eval-ensemble-size 1 --no-shuffle-eval

# srun python -m dsrnngan.main --eval-short --num-samples 64000 --restart --records-folder /user/work/uz22147/tfrecords/457221781d3b7cc9 --model-config-path /user/home/uz22147/repos/downscaling-cgan/config/model_config_small_cl2000.yaml --output-suffix small-cl2000 --num-images 2900 --eval-ensemble-size 1 --no-shuffle-eval
# srun python -m dsrnngan.main --eval-short --num-samples 64000 --restart --records-folder /user/work/uz22147/tfrecords/b3aac14803cf0125 --model-config-path /user/home/uz22147/repos/downscaling-cgan/config/model_config_small_cl2000.yaml --output-suffix small-cl2000-tponly --num-images 2900 --eval-ensemble-size 1 --no-shuffle-eval

# srun python -m dsrnngan.main --eval-short --num-samples 64000 --restart --records-folder /user/work/uz22147/tfrecords/457221781d3b7cc9 --model-config-path /user/home/uz22147/repos/downscaling-cgan/config/model_config_small_cl2000_oversampled.yaml --output-suffix small-cl2000-oversampled --num-images 2900 --eval-ensemble-size 1 --no-shuffle-eval

# srun python -m dsrnngan.main --eval-short --num-samples 64000 --restart --records-folder /user/work/uz22147/tfrecords/63d75ad191a59b10 --model-config-path /user/home/uz22147/repos/downscaling-cgan/config/model_config_small_cl500.yaml --output-suffix small-tponly-cl500-sqrt-norm --num-images 2900 --eval-ensemble-size 1 --no-shuffle-eval
# srun python -m dsrnngan.main --eval-short --num-samples 64000 --restart --records-folder /user/work/uz22147/tfrecords/457221781d3b7cc9 --model-config-path /user/home/uz22147/repos/downscaling-cgan/config/model_config_small_cl2000_batch_norm.yaml --output-suffix small-cl2000-batch-norm --num-images 2900 --eval-ensemble-size 1 --no-shuffle-eval


# srun python -m dsrnngan.main --eval-short --num-samples 64000 --restart --records-folder /user/work/uz22147/tfrecords/a4822ed35a44475d --model-config-path /user/home/uz22147/repos/downscaling-cgan/config/model_config_small_cl200_nocrop.yaml --output-suffix small-cl200-nologs-nocrop --num-images 2900 --eval-ensemble-size 1 --no-shuffle-eval --save-generated-samples;
# srun python -m scripts.make_plots --nickname small-cl100-nologs-nocrop --log-folder /user/work/uz22147/logs/cgan/a4822ed35a44475d_small-cl100-nologs-nocrop --model-number 64000

# srun python -m dsrnngan.main --eval-short --num-samples 64000 --restart --records-folder /user/work/uz22147/tfrecords/a4822ed35a44475d --model-config-path /user/home/uz22147/repos/downscaling-cgan/config/model_config_small_cl100.yaml --output-suffix small-cl100-nologs --num-images 2900 --eval-ensemble-size 1 --no-shuffle-eval --save-generated-samples;

# Main model run
# srun python -m scripts.make_plots --nickname small-cl100-nologs --log-folder /user/work/uz22147/logs/cgan/a4822ed35a44475d_small-cl100-nologs --model-number 64000

srun python -m dsrnngan.main --eval-blitz --num-samples 320000 --restart --records-folder /user/work/uz22147/tfrecords/998e5f0f54106b35 --model-config-path /user/home/uz22147/repos/downscaling-cgan/config/model_config_medium-cl100_crop100.yaml --output-suffix medium-cl100-nologs-crop100 --num-images 2900 --eval-ensemble-size 1 --no-shuffle-eval;
# srun python -m dsrnngan.main --eval-short --num-samples 320000 --restart --records-folder /user/work/uz22147/tfrecords/ec66788090f69890 --model-config-path /user/home/uz22147/repos/downscaling-cgan/config/model_config_medium_cl0.5_nocrop_LVonly.yaml --output-suffix medium-cl0.5-lvonly --num-images 2900 --eval-ensemble-size 1 --no-shuffle-eval;
# srun python -m dsrnngan.main --eval-short --num-samples 320000 --restart --records-folder /user/work/uz22147/tfrecords/998e5f0f54106b35 --model-config-path /user/home/uz22147/repos/downscaling-cgan/config/model_config_medium-cl50-nologs-nocrop.yaml --output-suffix medium-cl50-nologs-nocrop-seplake --num-images 2900 --eval-ensemble-size 1 --no-shuffle-eval;

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

