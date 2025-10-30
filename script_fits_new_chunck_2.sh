#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=pmt_doubleGauss
#SBATCH --output=pmt_doubleGauss_%A_%a.out
#SBATCH --error=pmt_doubleGauss_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-15   # Ajusta según total PMTs / chunk-size

#----------------- LOAD ENVIRONMENT -----------------
source /scratch/elena/elena_wcsim/build/env_wcsim.sh


#----------------- RUN PYTHON SCRIPT -----------------
python3 /scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/NEW_script_fits_chunck_2.py \
    --chunk-id ${SLURM_ARRAY_TASK_ID} \
    --chunk-size 100
