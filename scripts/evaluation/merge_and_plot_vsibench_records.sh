#!/usr/bin/env bash
set -euo pipefail

# Merge multiple VsiBench records files (possibly from different model types)
# and plot combined trend figures in one place.
#
# Metric computation examples (run separately, one MODEL_TYPE each run):
# 1) custom-spatial-mllm
# MODEL_TYPE=custom-spatial-mllm MODEL_DIRS_CSV='run_a,run_b' \
# bash scripts/evaluation/evaluate_vsibench_checkpoints_trend.sh
#
# 2) qwen2.5-vl
# MODEL_TYPE=qwen2.5-vl MODEL_DIRS_CSV='run_c,run_d' \
# bash scripts/evaluation/evaluate_vsibench_checkpoints_trend.sh
#
# 3) qwen3-vl
# MODEL_TYPE=qwen3-vl MODEL_DIRS_CSV='run_e,run_f' \
# bash scripts/evaluation/evaluate_vsibench_checkpoints_trend.sh
#
# Then put records file paths into RECORDS_FILES_CSV below and run this script.

#############################
# User Editable Config Zone #
#############################

# Comma-separated absolute paths to records_nframe*.txt generated from different runs.
RECORDS_FILES_CSV="/path/to/run1/records_nframe16.txt,/path/to/run2/records_nframe16.txt"

# Frame settings to plot. One merged file will be produced per nframe.
NFRAMES_LIST="16"

# Plot metrics, same format as vsibench_trend_plot_utils.py
PLOT_METRICS_CSV="all:micro,all:macro"

# Output directory for merged records and combined plots.
OUT_DIR="/tmp/vsibench_multi_modeltype_combined"

PLOT_UTIL_PY="scripts/evaluation/vsibench_trend_plot_utils.py"

##################
# Helper methods #
##################

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

csv_to_array() {
  local csv="$1"
  local -n out_arr="$2"
  out_arr=()
  if [[ -z "${csv// }" ]]; then
    return 0
  fi
  IFS=',' read -r -a out_arr <<< "$csv"
}

################
# Main routine #
################

cd "$(dirname "$0")/../.."
mkdir -p "$OUT_DIR"

csv_to_array "$RECORDS_FILES_CSV" RECORDS_FILES
read -r -a NFRAMES <<< "$NFRAMES_LIST"

if [[ ${#RECORDS_FILES[@]} -lt 2 ]]; then
  log "Need at least 2 records files to merge. Current count=${#RECORDS_FILES[@]}"
  exit 1
fi

if [[ ${#NFRAMES[@]} -eq 0 ]]; then
  log "NFRAMES_LIST is empty."
  exit 1
fi

for nframe in "${NFRAMES[@]}"; do
  merged_records="${OUT_DIR}/records_merged_nframe${nframe}.txt"
  : > "$merged_records"

  # JJ: merge all existing records files for this nframe, deduplicate same line, then plot combined trends.
  for rf in "${RECORDS_FILES[@]}"; do
    if [[ -f "$rf" ]]; then
      cat "$rf" >> "$merged_records"
    else
      log "[WARN] Missing records file: $rf"
    fi
  done

  if [[ ! -s "$merged_records" ]]; then
    log "No records merged for nframe=${nframe}. Skip plotting."
    continue
  fi

  # Deduplicate exact duplicate lines while keeping deterministic order.
  dedup_tmp="${merged_records}.dedup"
  sort -u "$merged_records" > "$dedup_tmp"
  mv "$dedup_tmp" "$merged_records"

  out_prefix="${OUT_DIR}/combined-nframe${nframe}"
  python "$PLOT_UTIL_PY" \
    --mode combined \
    --records_file "$merged_records" \
    --nframe "$nframe" \
    --out_prefix "$out_prefix" \
    --plot_metrics "$PLOT_METRICS_CSV"

  log "Merged records: $merged_records"
  log "Combined plots prefix: ${out_prefix}"
done

log "Done."
