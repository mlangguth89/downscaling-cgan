#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-20:00:00
#SBATCH --mem=10gb
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=monthly-cat
#SBATCH --partition short
#SBATCH --array=2003
#SBATCH --output=logs/slurm-%A_%a.out

source ~/.bashrc
module load apps/nco-toolkit/4.9.2-gcc
module load lang/cdo/1.9.8-gcc

for month_str in 01 02 03 04 05 06 07 08 09 10 11 12
do
    infile="/bp1/geog-tropical/users/uz22147/east_africa_data/IMERG/half_hourly/final/3B-HHR.MS.MRG.3IMERG.${SLURM_ARRAY_TASK_ID}${month_str}*"
    tmp_file="/bp1/geog-tropical/users/uz22147/east_africa_data/IMERG/monthly/${SLURM_ARRAY_TASK_ID}${month_str}_tmp.nc"
    outfile="/bp1/geog-tropical/users/uz22147/east_africa_data/IMERG/monthly/${SLURM_ARRAY_TASK_ID}${month_str}_total.nc"

    srun cdo cat ${infile} ${tmp_file}; cdo -O -b F32 monsum ${tmp_file} ${outfile}; rm ${tmp_file};
done

