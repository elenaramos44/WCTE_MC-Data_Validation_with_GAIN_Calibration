#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=script_fits
#SBATCH --output=script_fits_%A_%a.out
#SBATCH --error=script_fits_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-1    # adjust based on total PMTs/100


source /scratch/elena/elena_wcsim/build/env_wcsim.sh


source /scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/pyarrow_env/bin/activate

#debug
which python
python -m site


python /scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/script_fit_LED.py \
    --chunk-id ${SLURM_ARRAY_TASK_ID} \
    --chunk-size 100 \
    --folder /dipc/elena/WCTE_2025_commissioning/parquet_files \
    --run-number led_scan_calibration_20250404145411_hv_0_20250404145412_7



