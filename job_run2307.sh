#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=save_pmt_waveforms
#SBATCH --output=save_pmt_waveforms_%A_%a.out
#SBATCH --error=save_pmt_waveforms_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --array=0-23                # 24 ROOT files
#SBATCH --mail-type=END,FAIL        
#SBATCH --mail-user=elena.ramos@dipc.org

source /scratch/elena/elena_wcsim/build/env_wcsim.sh

#use Python packages from /scratch
export PYTHONPATH=/scratch/elena/python_packages:$PYTHONPATH

ROOT_DIR=/scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/root_files
OUT_DIR=/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307

ROOT_FILES=($(ls $ROOT_DIR/WCTE_offline_R2307S0P*.root | sort))


ROOT_FILE=${ROOT_FILES[$SLURM_ARRAY_TASK_ID]}
echo "Processing ROOT file: $ROOT_FILE"

#run Python script
python3 /scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/save_pmt_waveforms.py \
    --run 2307 \
    --outdir $OUT_DIR \
    --root-file $ROOT_FILE
