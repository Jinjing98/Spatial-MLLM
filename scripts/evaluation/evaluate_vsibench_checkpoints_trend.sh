#!/usr/bin/env bash
set -euo pipefail

# Isolated checkpoint-trend evaluation script for VsiBench.
# - Searches model run directories.
# - Evaluates checkpoint-* models.
# - Plots trend curves with x-axis as checkpoint step number.
#
# Direct plotting utility example (single model):
# python scripts/evaluation/vsibench_trend_plot_utils.py \
#   --mode single \
#   --records_file /data/.../checkpoint_trends/records_nframe16.txt \
#   --nframe 16 \
#   --out_prefix /data/.../checkpoint_trends/20260301_135514_spatial-mllm-sft_baseline_sp133krp2k-nframe16 \
#   --model_root_name 20260301_135514_spatial-mllm-sft_baseline_sp133krp2k

#############################
# User Editable Config Zone #
#############################
# Edit these variables directly in this file for stable experiment records.

DATA_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets"
SFT_MODELS_ROOT="/data/horse/ws/jixu233b-metadata_ws/exps/train/spatialmllm"
EVAL_RESULTS_BASE="/data/horse/ws/jixu233b-metadata_ws/exps/stats/spatialmllm_results/results_trends"

MODEL_TYPE="custom-spatial-mllm"
MODEL_SEARCH_PATTERN="20260301_135514_spatial-mllm-sft_baseline_sp133krp2k"
# will ignore MODEL_SEARCH_PATTERN if MODEL_DIRS_CSV is not empty
# JJ:
MODEL_DIRS_CSV=""                # optional concrete model dirs, comma separated; entry can be abs path or dir name under SFT_MODELS_ROOT
MODEL_DIRS_CSV="20260301_135514_spatial-mllm-sft_baseline_sp133krp2k,20260301_121711_spatial-mllm-sft_baseline_skipCnc_sp133krp2k"
PLOT_METRICS_CSV="all:micro,all:macro"  # e.g. "all:micro,all:macro,acc:micro,mra:macro"
CKPT_STEPS_CSV="528,8446"                # comma separated checkpoint step list, e.g. "1000,2000"; empty => all found
DATASETS_CSV="arkitscenes"       # comma separated
QUESTION_TYPES_CSV="object_rel_distance" # comma separated
SCENE_NAMES_CSV="42446103"       # comma separated; empty string => all scenes


MODEL_NAME_SUFFIX="-baseline-sp133krp2k"

RUN_EVAL=1                       # 1: run eval, 0: only read existing metrics for plotting
EVAL_ALL_CHECKPOINTS=1
INCLUDE_FINAL_MODEL=0            # final model root optional; x-axis is checkpoint step
PLOT_TREND=1
PLOT_COMBINED=1

SAMPLING="mergeaware_sa_sampling"
MERGEAWARE_DETAILS="_rnd_idxss1"
NFRAMES_LIST="16"                # space separated, e.g. "8 16 32"


EVAL_PY="src/evaluation/vsibench/eval_vsibench.py"
PLOT_UTIL_PY="scripts/evaluation/vsibench_trend_plot_utils.py"
OUTPUT_ROOT="${EVAL_RESULTS_BASE}/vsibench_${SAMPLING}"
EVAL_OUTPUT_DIR="${OUTPUT_ROOT}/evals"
TREND_DIR="${OUTPUT_ROOT}/checkpoint_trends"

##################
# Helper methods #
##################

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

join_by() {
  local IFS="$1"
  shift
  echo "$*"
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

array_contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

remove_array_item() {
  local needle="$1"
  local -n arr_ref="$2"
  local -a kept=()
  local item
  for item in "${arr_ref[@]}"; do
    if [[ "$item" != "$needle" ]]; then
      kept+=("$item")
    fi
  done
  arr_ref=("${kept[@]}")
}

discover_model_roots() {
  local pattern="$1"
  local root="$2"
  local -n out_arr="$3"
  out_arr=()
  while IFS= read -r p; do
    out_arr+=("$p")
  done < <(find "$root" -maxdepth 1 -mindepth 1 -type d -name "$pattern" | sort -V)
}

discover_model_roots_from_csv() {
  local model_dirs_csv="$1"
  local root="$2"
  local -n out_arr="$3"
  out_arr=()

  local -a requested=()
  csv_to_array "$model_dirs_csv" requested

  local model_item
  local resolved
  for model_item in "${requested[@]}"; do
    if [[ -z "${model_item// }" ]]; then
      continue
    fi
    if [[ "$model_item" = /* ]]; then
      resolved="$model_item"
    else
      resolved="${root}/${model_item}"
    fi
    if [[ -d "$resolved" ]]; then
      out_arr+=("$resolved")
    else
      log "[WARN] Requested model not found: ${model_item} (resolved: ${resolved})"
    fi
  done
}

discover_checkpoint_paths() {
  local model_root="$1"
  local -n out_arr="$2"
  out_arr=()

  if [[ "$EVAL_ALL_CHECKPOINTS" == "1" ]]; then
    while IFS= read -r p; do
      out_arr+=("$p")
    done < <(find "$model_root" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V)
  else
    out_arr+=("$model_root")
  fi

  if [[ "$INCLUDE_FINAL_MODEL" == "1" ]]; then
    out_arr+=("$model_root")
  fi
}

run_one_eval() {
  local ckpt_path="$1"
  local run_label="$2"
  local nframe="$3"
  local datasets_suffix="$4"
  local questions_suffix="$5"
  local scene_names_str="$6"
  local exp_dir="$7"

  local log_file="${exp_dir}/run.log"
  mkdir -p "$exp_dir"

  log "Eval model=${run_label} nframe=${nframe}"

  local -a extra_scene_args=()
  if [[ -n "$scene_names_str" ]]; then
    # shellcheck disable=SC2206
    local scene_arr=( $scene_names_str )
    extra_scene_args=(--scene_names "${scene_arr[@]}")
  fi

  # shellcheck disable=SC2086
  python "$EVAL_PY" \
    --model_path "$ckpt_path" \
    --model_type "$MODEL_TYPE" \
    --nframes "$nframe" \
    --annotation_dir "${DATA_ROOT}/vsibench" \
    --question_types ${QUESTION_TYPES[@]} \
    --datasets ${DATASETS[@]} \
    --video_dir "${DATA_ROOT}/vsibench/${SAMPLING}_${nframe}f${MERGEAWARE_DETAILS}" \
    --batch_size 1 \
    --output_dir "$exp_dir" \
    --output_name "eval_result" \
    "${extra_scene_args[@]}" \
    2>&1 | tee -a "$log_file"

  echo "$exp_dir"
}

append_record() {
  local records_file="$1"
  local model_root_name="$2"
  local ckpt_step="$3"
  local metrics_path="$4"
  printf '%s|%s|%s\n' "$model_root_name" "$ckpt_step" "$metrics_path" >> "$records_file"
}

plot_one_model_trend() {
  local model_root_name="$1"
  local nframe="$2"
  local records_file="$3"
  local out_prefix="$4"
  python "$PLOT_UTIL_PY" \
    --mode single \
    --records_file "$records_file" \
    --nframe "$nframe" \
    --out_prefix "$out_prefix" \
    --model_root_name "$model_root_name" \
    --plot_metrics "$PLOT_METRICS_CSV"
}

plot_combined_trend() {
  local nframe="$1"
  local records_file="$2"
  local out_prefix="$3"
  python "$PLOT_UTIL_PY" \
    --mode combined \
    --records_file "$records_file" \
    --nframe "$nframe" \
    --out_prefix "$out_prefix" \
    --plot_metrics "$PLOT_METRICS_CSV"
}

################
# Main routine #
################

cd "$(dirname "$0")/../.."

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$EVAL_OUTPUT_DIR"
mkdir -p "$TREND_DIR"

RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_CONFIG_FILE="${TREND_DIR}/run_config_${RUN_STAMP}.txt"
cat > "$RUN_CONFIG_FILE" <<EOF
RUN_STAMP=${RUN_STAMP}
DATA_ROOT=${DATA_ROOT}
SFT_MODELS_ROOT=${SFT_MODELS_ROOT}
EVAL_RESULTS_BASE=${EVAL_RESULTS_BASE}
MODEL_TYPE=${MODEL_TYPE}
MODEL_SEARCH_PATTERN=${MODEL_SEARCH_PATTERN}
MODEL_DIRS_CSV=${MODEL_DIRS_CSV}
MODEL_NAME_SUFFIX=${MODEL_NAME_SUFFIX}
RUN_EVAL=${RUN_EVAL}
EVAL_ALL_CHECKPOINTS=${EVAL_ALL_CHECKPOINTS}
INCLUDE_FINAL_MODEL=${INCLUDE_FINAL_MODEL}
PLOT_TREND=${PLOT_TREND}
PLOT_COMBINED=${PLOT_COMBINED}
PLOT_METRICS_CSV=${PLOT_METRICS_CSV}
SAMPLING=${SAMPLING}
MERGEAWARE_DETAILS=${MERGEAWARE_DETAILS}
NFRAMES_LIST=${NFRAMES_LIST}
CKPT_STEPS_CSV=${CKPT_STEPS_CSV}
DATASETS_CSV=${DATASETS_CSV}
QUESTION_TYPES_CSV=${QUESTION_TYPES_CSV}
SCENE_NAMES_CSV=${SCENE_NAMES_CSV}
EVAL_PY=${EVAL_PY}
PLOT_UTIL_PY=${PLOT_UTIL_PY}
OUTPUT_ROOT=${OUTPUT_ROOT}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}
TREND_DIR=${TREND_DIR}
EOF
log "Run config saved: $RUN_CONFIG_FILE"

csv_to_array "$DATASETS_CSV" DATASETS
csv_to_array "$QUESTION_TYPES_CSV" QUESTION_TYPES
csv_to_array "$SCENE_NAMES_CSV" SCENE_NAMES
csv_to_array "$CKPT_STEPS_CSV" REQUESTED_CKPT_STEPS

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  log "DATASETS_CSV is empty. Set at least one dataset."
  exit 1
fi
if [[ ${#QUESTION_TYPES[@]} -eq 0 ]]; then
  log "QUESTION_TYPES_CSV is empty. Set at least one question type."
  exit 1
fi
if [[ ${#REQUESTED_CKPT_STEPS[@]} -gt 0 ]]; then
  log "Checkpoint filter enabled. Will evaluate ckpt steps: $(join_by "," "${REQUESTED_CKPT_STEPS[@]}")"
fi

DATASET_SUFFIX="_$(join_by "_" "${DATASETS[@]}")"
QUESTION_SUFFIX="_$(join_by "_" "${QUESTION_TYPES[@]}")"
SCENE_NAMES_STR="$(join_by " " "${SCENE_NAMES[@]}")"

log "Searching models in: $SFT_MODELS_ROOT"
if [[ -n "${MODEL_DIRS_CSV// }" ]]; then
  log "MODEL_DIRS_CSV=$MODEL_DIRS_CSV"
  discover_model_roots_from_csv "$MODEL_DIRS_CSV" "$SFT_MODELS_ROOT" MODEL_ROOTS
else
  log "MODEL_SEARCH_PATTERN=$MODEL_SEARCH_PATTERN"
  discover_model_roots "$MODEL_SEARCH_PATTERN" "$SFT_MODELS_ROOT" MODEL_ROOTS
fi

if [[ ${#MODEL_ROOTS[@]} -eq 0 ]]; then
  log "No model run directories matched."
  exit 1
fi

log "Found ${#MODEL_ROOTS[@]} model run(s)."
for m in "${MODEL_ROOTS[@]}"; do
  log "  - $m"
done

read -r -a NFRAMES <<< "$NFRAMES_LIST"
if [[ ${#NFRAMES[@]} -eq 0 ]]; then
  log "NFRAMES_LIST is empty."
  exit 1
fi

for nframe in "${NFRAMES[@]}"; do
  records_file="${TREND_DIR}/records_nframe${nframe}.txt"
  : > "$records_file"

  for model_root in "${MODEL_ROOTS[@]}"; do
    model_root_name="$(basename "$model_root")"
    discover_checkpoint_paths "$model_root" CKPT_PATHS

    if [[ ${#CKPT_PATHS[@]} -eq 0 ]]; then
      log "No checkpoints found for $model_root_name"
      continue
    fi

    log "Model $model_root_name: ${#CKPT_PATHS[@]} candidate path(s)"
    unmatched_steps=("${REQUESTED_CKPT_STEPS[@]}")

    for ckpt_path in "${CKPT_PATHS[@]}"; do
      ckpt_base="$(basename "$ckpt_path")"
      if [[ "$ckpt_base" == checkpoint-* ]]; then
        ckpt_step="${ckpt_base#checkpoint-}"
        run_label="${MODEL_TYPE}${MODEL_NAME_SUFFIX}-${model_root_name}-ckpt${ckpt_step}"
        ckpt_group="ckpt${ckpt_step}"
      else
        ckpt_step="final"
        run_label="${MODEL_TYPE}${MODEL_NAME_SUFFIX}-${model_root_name}-final"
        ckpt_group="final"
      fi

      if [[ ${#REQUESTED_CKPT_STEPS[@]} -gt 0 ]] && ! array_contains "$ckpt_step" "${REQUESTED_CKPT_STEPS[@]}"; then
        continue
      fi
      if [[ ${#REQUESTED_CKPT_STEPS[@]} -gt 0 ]]; then
        remove_array_item "$ckpt_step" unmatched_steps
      fi

      exp_dir="${EVAL_OUTPUT_DIR}/${model_root_name}/${ckpt_group}/${nframe}f${DATASET_SUFFIX}${QUESTION_SUFFIX}"
      metrics_path="${exp_dir}/eval_result/metrics_${MODEL_TYPE}.json"

      if [[ "$RUN_EVAL" == "1" ]]; then
        run_one_eval "$ckpt_path" "$run_label" "$nframe" "$DATASET_SUFFIX" "$QUESTION_SUFFIX" "$SCENE_NAMES_STR" "$exp_dir" >/dev/null
      fi

      if [[ -f "$metrics_path" ]]; then
        append_record "$records_file" "$model_root_name" "$ckpt_step" "$metrics_path"
      else
        log "[WARN] Missing metrics: $metrics_path"
      fi
    done

    if [[ ${#unmatched_steps[@]} -gt 0 ]]; then
      for step in "${unmatched_steps[@]}"; do
        log "[WARN] Model ${model_root_name}: no checkpoint matched requested step '${step}'"
      done
    fi
  done

  if [[ "$PLOT_TREND" == "1" ]]; then
    for model_root in "${MODEL_ROOTS[@]}"; do
      model_root_name="$(basename "$model_root")"
      out_prefix="${TREND_DIR}/${model_root_name}-nframe${nframe}"
      plot_one_model_trend "$model_root_name" "$nframe" "$records_file" "$out_prefix"

      log "Manual single-model replot command:"
      log "python ${PLOT_UTIL_PY} --mode single --records_file ${records_file} --nframe ${nframe} --out_prefix ${out_prefix} --model_root_name ${model_root_name}"
    done

    if [[ "$PLOT_COMBINED" == "1" ]]; then
      plot_combined_trend "$nframe" "$records_file" "${TREND_DIR}/combined-nframe${nframe}"

      log "Manual combined replot command:"
      log "python ${PLOT_UTIL_PY} --mode combined --records_file ${records_file} --nframe ${nframe} --out_prefix ${TREND_DIR}/combined-nframe${nframe}"
    fi
  fi

  log "Finished nframe=${nframe}. Trend outputs in: $TREND_DIR"
done

log "All done."
