#!/bin/bash
#!/bin/bash
#SBATCH --job-name=merge_waveforms
#SBATCH --output=merge_waveforms_%A_%a.out
#SBATCH --error=merge_waveforms_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-74   # example: 15k PMTs / 200 PMTs per job = 75 jobs

module purge
module load Python/3.7.4-GCCcore-8.3.0

FOLDER="/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307/waveforms_including_position"
PMT_JSON="$FOLDER/pmts_list.json"

PMTS_PER_JOB=200

START=$(( SLURM_ARRAY_TASK_ID * PMTS_PER_JOB ))
END=$(( START + PMTS_PER_JOB - 1 ))

python3 /scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/MERGE.py \
    --folder $FOLDER \
    --pmt-json $PMT_JSON \
    --start $START \
    --end $END
