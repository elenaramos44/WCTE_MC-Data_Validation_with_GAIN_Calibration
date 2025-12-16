#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=mPMTfit_2307
#SBATCH --output=mPMTfit_2307_%A_%a.out
#SBATCH --error=mPMTfit_2307_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-94

#----------------- LOAD ENVIRONMENT -----------------
module purge
module load matplotlib/3.5.2-foss-2022a   # trae Python + numpy + scipy + matplotlib

#----------------- RUN PYTHON SCRIPT -----------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python /scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/mPMT_fit_script.py
