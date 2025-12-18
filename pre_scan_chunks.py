#!/usr/bin/env python3
import uproot
import glob
import math
import json

# ---------- USER PARAMETERS ----------
RUN = 2307
CHUNK_SIZE = 250
TREE_NAME = "WCTEReadoutWindows"
ROOT_DIR = "/scratch/elena/WCTE_DATA_ANALYSIS/WCTE_MC-Data_Validation_with_GAIN_Calibration/root_files"
EXCLUDE_PARTS = [0, 1]   # ya procesados
OUTPUT_JSON = "chunks_per_part.json"
# -------------------------------------

parts = {}

for fname in sorted(glob.glob(f"{ROOT_DIR}/WCTE_offline_R{RUN}S0P*.root")):
    part = int(fname.split("P")[-1].split(".")[0])
    if part in EXCLUDE_PARTS:
        continue

    with uproot.open(f"{fname}:{TREE_NAME}") as tree:
        n_events = tree.num_entries

    n_chunks = math.ceil(n_events / CHUNK_SIZE)
    parts[part] = n_chunks
    print(f"Part {part}: {n_events} events → {n_chunks} chunks")

# Guardar en JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(parts, f, indent=2)

print(f"\nPre-scan completo. Resultados guardados en {OUTPUT_JSON}")
