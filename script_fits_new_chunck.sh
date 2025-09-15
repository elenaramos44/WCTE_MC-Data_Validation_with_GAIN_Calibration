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
#SBATCH --array=0-15    # adjust based on total PMTs/100

# Load environment
source /scratch/elena/elena_wcsim/build/env_wcsim.sh

# Run Python script with the array index as chunk-id
python3 /scratch/elena/WCTE_2025_commissioning/2025_data/WCTE_BRB_Data_Analysis/script_fits_new_chunck.py \
    --chunk-id ${SLURM_ARRAY_TASK_ID} --chunk-size 100
