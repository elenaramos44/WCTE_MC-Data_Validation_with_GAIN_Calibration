#!/bin/bash
#!/bin/bash
#SBATCH --job-name=merge_wf_run2307_p1
#SBATCH --output=merge_wf_run2307_p1_%A_%a.out
#SBATCH --error=merge_wf_run2307_p1_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-75   # example: 1567 PMTs / 200 PMTs per job = 75 jobs

module purge
module load Python/3.7.4-GCCcore-8.3.0

FOLDER="/scratch/elena/WCTE_DATA_ANALYSIS/waveform_npz/run2307/waveforms_including_position"
PMT_JSON="$FOLDER/pmts_list.json"

PMTS_PER_JOB=200

START=$(( SLURM_ARRAY_TASK_ID * PMTS_PER_JOB ))
END=$(( START + PMTS_PER_JOB - 1 ))

python3 /scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/MERGE_separate_parts.py \
    --folder $FOLDER \
    --pmt-json $PMT_JSON \
    --start $START \
    --end $END
