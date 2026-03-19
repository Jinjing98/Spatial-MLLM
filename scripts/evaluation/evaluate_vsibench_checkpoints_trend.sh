#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1 #2
#SBATCH --gres=gpu:4 #1           # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --gpus-per-task=4 #1
#SBATCH --time=35:00:00
#SBATCH --mem=80G
#SBATCH --partition=capella
#SBATCH --mail-user=xvjinjing8@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --error=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.err
#SBATCH --output=/data/horse/ws/jixu233b-metadata_ws/hpc_out/%j.out

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
#
# Bash override examples (env vars override in-script defaults):
# 1) One-shot inline override:
# MODEL_TYPE=custom-spatial-mllm MODEL_SEARCH_PATTERN='20260301*baseline*' \
# bash scripts/evaluation/evaluate_vsibench_checkpoints_trend.sh
#
# 2) Explicitly use MODEL_DIRS_CSV (will ignore MODEL_SEARCH_PATTERN):
# MODEL_DIRS_CSV='run_a,run_b' \
# bash scripts/evaluation/evaluate_vsibench_checkpoints_trend.sh
#
# 3) Force pattern mode by setting MODEL_DIRS_CSV empty:
# MODEL_DIRS_CSV='' MODEL_SEARCH_PATTERN='20260301_135514*' \
# bash scripts/evaluation/evaluate_vsibench_checkpoints_trend.sh
#
# 4) Export style override for current shell session:
# export MODEL_TYPE=custom-spatial-mllm
# export MODEL_DIRS_CSV='run_a,run_b'
# bash scripts/evaluation/evaluate_vsibench_checkpoints_trend.sh

# JJ: follow existing HPC bootstrap style used in other evaluation scripts.
source /software/rapids/r24.10/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /data/horse/ws/jixu233b-3d_ws/envs/spatial-mllm
module load release/24.04
module load CUDA/12.4.0
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
mkdir -p $TRITON_CACHE_DIR

# JJ: align run-directory handling with other eval scripts.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
fi

#############################
# User Editable Config Zone #
#############################
# Edit these variables directly in this file for stable experiment records.

# Part 1/4: 数据与模型根路径（定义评估数据来源、模型来源、结果总根目录）
DATA_ROOT="/data/horse/ws/jixu233b-metadata_ws/datasets"
SFT_MODELS_ROOT="/data/horse/ws/jixu233b-metadata_ws/exps/train/spatialmllm"
EVAL_RESULTS_BASE="/data/horse/ws/jixu233b-metadata_ws/exps/stats/spatialmllm_results/results_trends"

# Part 2/4: 评估对象与筛选范围（定义模型选择、checkpoint 选择、数据子集选择）
MODEL_TYPE="${MODEL_TYPE-custom-spatial-mllm-lvsm}"  # e.g. "custom-spatial-mllm-lvsm", "qwen2.5-vl", "qwen3-vl"; used for both model search and eval config
# Supports one or multiple patterns (comma separated), e.g.
# "20260301*baseline*,20260302*ablation*"
MODEL_SEARCH_PATTERN="${MODEL_SEARCH_PATTERN-*141138*,*223914*}"
# will ignore MODEL_SEARCH_PATTERN if MODEL_DIRS_CSV is not empty
# JJ: model selection vars support both external env override and in-script default edits.
MODEL_DIRS_CSV="${MODEL_DIRS_CSV-}"  # optional concrete model dirs, comma separated; entry can be abs path or dir name under SFT_MODELS_ROOT
# JJ: CKPT_STEPS_CSV 支持「脚本内默认 + Bash 外部覆盖」；填写如 "1000,2000,3000"，留空 "" 表示评估全部 checkpoints。
# Bash 传参示例: CKPT_STEPS_CSV='1000,2000' bash scripts/evaluation/evaluate_vsibench_checkpoints_trend.sh
# Bash 全量示例: CKPT_STEPS_CSV='' bash scripts/evaluation/evaluate_vsibench_checkpoints_trend.sh
CKPT_STEPS_CSV="${CKPT_STEPS_CSV-528,8446}"                # comma separated checkpoint step list, e.g. "1000,2000"; empty => all found
DATASETS_CSV="arkitscenes"       # comma separated; empty string => all datasets
QUESTION_TYPES_CSV="object_rel_distance" # comma separated; empty string => all question types
SCENE_NAMES_CSV="42446103"       # comma separated; empty string => all scenes
MODEL_NAME_SUFFIX="-baseline-sp133krp2k"
MODEL_NAME_SUFFIX="-TEST-PLOT"

# Part 3/4: 评估执行与绘图行为开关（控制是否跑评估、是否画图、视频采样设置）
RUN_EVAL=1                       # 1: run eval, 0: only read existing metrics for plotting
EVAL_ALL_CHECKPOINTS=1
INCLUDE_FINAL_MODEL=0            # final model root optional; x-axis is checkpoint step
PLOT_TREND=1
PLOT_COMBINED=1
LIVE_PLOT_WHILE_EVAL=1           # 1: when RUN_EVAL=1, refresh trend plots after each new metrics point
PLOT_METRICS_CSV="all:micro,all:macro,acc:micro,mra:macro"  # e.g. "all:micro,all:macro,acc:micro,mra:macro"
# SAMPLING="mergeaware_sa_sampling"
# MERGEAWARE_DETAILS="_rnd_idxss1"
SAMPLING="uniform_sampling"
MERGEAWARE_DETAILS=""
NFRAMES_LIST="16"                # space separated, e.g. "8 16 32"

# Part 4/4: 脚本入口与输出目录组织（定义调用脚本路径与结果落盘路径）
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

show_eval_progress() {
  local done="$1"
  local total="$2"
  local nframe="$3"
  if [[ "$total" -le 0 ]]; then
    return 0
  fi
  local percent
  percent="$(awk -v d="$done" -v t="$total" 'BEGIN { printf "%.1f", (d * 100.0) / t }')"
  printf '\r[EvalProgress][nframe=%s] %d/%d (%s%%)' "$nframe" "$done" "$total" "$percent"
  if [[ "$done" -ge "$total" ]]; then
    printf '\n'
  fi
}

discover_model_roots() {
  local pattern="$1"
  local root="$2"
  local -n out_arr="$3"
  out_arr=()
  local -a patterns=()
  local -a raw_matches=()
  local pat

  csv_to_array "$pattern" patterns
  if [[ ${#patterns[@]} -eq 0 ]]; then
    return 0
  fi

  # JJ: support comma-separated model search patterns; merge matches, deduplicate, keep version sort.
  for pat in "${patterns[@]}"; do
    if [[ -z "${pat// }" ]]; then
      continue
    fi
    while IFS= read -r p; do
      raw_matches+=("$p")
    done < <(find "$root" -maxdepth 1 -mindepth 1 -type d -name "$pat")
  done

  if [[ ${#raw_matches[@]} -eq 0 ]]; then
    return 0
  fi

  while IFS= read -r p; do
    out_arr+=("$p")
  done < <(printf '%s\n' "${raw_matches[@]}" | sort -Vu)
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

sanitize_token() {
  local raw="$1"
  # Keep filename-safe chars only.
  echo "$raw" | tr '[:space:]/:' '_' | tr -cd '[:alnum:]_.-'
}

short_model_label() {
  local raw="$1"
  local safe
  local short
  safe="$(sanitize_token "$raw")"

  # JJ: shorten long model names for filesystem-safe plot filenames (prefer YYYYMMDD_HHMMSS_<name> pattern).
  if [[ "$safe" =~ ^([0-9]{8}_[0-9]{6}_[^_]+) ]]; then
    short="${BASH_REMATCH[1]}"
  else
    IFS='_' read -r f1 f2 f3 _ <<< "$safe"
    if [[ -n "${f1:-}" && -n "${f2:-}" && -n "${f3:-}" ]]; then
      short="${f1}_${f2}_${f3}"
    else
      short="$safe"
    fi
  fi

  # Keep a hard cap to avoid Errno 36 even when pattern extraction fails.
  echo "${short:0:48}"
}

build_models_tag() {
  local -n model_roots_ref="$1"
  local -a names=()
  local m
  for m in "${model_roots_ref[@]}"; do
    names+=("$(short_model_label "$(basename "$m")")")
  done
  if [[ ${#names[@]} -eq 0 ]]; then
    echo "nomodel"
    return 0
  fi
  if [[ ${#names[@]} -eq 1 ]]; then
    echo "${names[0]}"
    return 0
  fi
  # JJ: multi-model naming for combined plots; include first 2 names + model count to keep filename readable.
  echo "${names[0]}__${names[1]}__n${#names[@]}models"
}

check_model_type_consistency() {
  local model_root="$1"
  local model_root_name="$2"
  local config_path="${model_root}/config.json"
  local train_model_type=""

  # JJ: best-effort consistency check; warn on mismatch/missing config, but always keep current eval MODEL_TYPE (Bash/script value).
  if [[ ! -f "$config_path" ]]; then
    log "[WARN] Model ${model_root_name}: config.json not found at ${config_path}. Skip model_type consistency check; use MODEL_TYPE=${MODEL_TYPE}."
    return 0
  fi

  train_model_type="$(python - "$config_path" <<'PY'
import json
import sys

cfg = sys.argv[1]
try:
    with open(cfg, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    print("__PARSE_ERROR__")
    raise SystemExit(0)

for key in ("model_type", "model_name_or_path", "model_name", "model"):
    value = data.get(key)
    if isinstance(value, str) and value.strip():
        print(value.strip())
        break
else:
    print("")
PY
)"

  if [[ "$train_model_type" == "__PARSE_ERROR__" ]]; then
    log "[WARN] Model ${model_root_name}: failed to parse ${config_path}. Skip model_type consistency check; use MODEL_TYPE=${MODEL_TYPE}."
    return 0
  fi

  if [[ -z "$train_model_type" ]]; then
    log "[WARN] Model ${model_root_name}: no model_type-like field found in ${config_path}. Use MODEL_TYPE=${MODEL_TYPE}."
    return 0
  fi

  if [[ "$train_model_type" != "$MODEL_TYPE" ]]; then
    log "[WARN] Model ${model_root_name}: config model type='${train_model_type}' but eval MODEL_TYPE='${MODEL_TYPE}'. Using eval MODEL_TYPE='${MODEL_TYPE}'."
  else
    log "Model ${model_root_name}: model_type check passed (${MODEL_TYPE})."
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
  local -a extra_dataset_args=()
  local -a extra_question_args=()
  if [[ -n "$scene_names_str" ]]; then
    # shellcheck disable=SC2206
    local scene_arr=( $scene_names_str )
    extra_scene_args=(--scene_names "${scene_arr[@]}")
  fi
  # JJ: only pass dataset/question filters when configured; empty means full-data eval.
  if [[ ${#DATASETS[@]} -gt 0 ]]; then
    extra_dataset_args=(--datasets "${DATASETS[@]}")
  fi
  if [[ ${#QUESTION_TYPES[@]} -gt 0 ]]; then
    extra_question_args=(--question_types "${QUESTION_TYPES[@]}")
  fi

  # shellcheck disable=SC2086
  python "$EVAL_PY" \
    --model_path "$ckpt_path" \
    --model_type "$MODEL_TYPE" \
    --nframes "$nframe" \
    --annotation_dir "${DATA_ROOT}/vsibench" \
    "${extra_question_args[@]}" \
    "${extra_dataset_args[@]}" \
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
LIVE_PLOT_WHILE_EVAL=${LIVE_PLOT_WHILE_EVAL}
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

# JJ: allow full-data/full-question eval when DATASETS_CSV / QUESTION_TYPES_CSV is empty.
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  log "DATASETS_CSV is empty. Will evaluate all datasets."
fi
if [[ ${#QUESTION_TYPES[@]} -eq 0 ]]; then
  log "QUESTION_TYPES_CSV is empty. Will evaluate all question types."
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
MODELS_TAG="$(build_models_tag MODEL_ROOTS)"
for m in "${MODEL_ROOTS[@]}"; do
  check_model_type_consistency "$m" "$(basename "$m")"
done

read -r -a NFRAMES <<< "$NFRAMES_LIST"
if [[ ${#NFRAMES[@]} -eq 0 ]]; then
  log "NFRAMES_LIST is empty."
  exit 1
fi

for nframe in "${NFRAMES[@]}"; do
  records_file="${TREND_DIR}/records_nframe${nframe}.txt"
  : > "$records_file"
  eval_total=0
  eval_done=0

  # JJ: pre-count selected checkpoints for online progress display when RUN_EVAL=1.
  if [[ "$RUN_EVAL" == "1" ]]; then
    for model_root in "${MODEL_ROOTS[@]}"; do
      discover_checkpoint_paths "$model_root" CKPT_PATHS_PROGRESS
      for ckpt_path in "${CKPT_PATHS_PROGRESS[@]}"; do
        ckpt_base_progress="$(basename "$ckpt_path")"
        if [[ "$ckpt_base_progress" == checkpoint-* ]]; then
          ckpt_step_progress="${ckpt_base_progress#checkpoint-}"
        else
          ckpt_step_progress="final"
        fi
        if [[ ${#REQUESTED_CKPT_STEPS[@]} -gt 0 ]] && ! array_contains "$ckpt_step_progress" "${REQUESTED_CKPT_STEPS[@]}"; then
          continue
        fi
        eval_total=$((eval_total + 1))
      done
    done
    log "nframe=${nframe} eval target checkpoints: ${eval_total}"
    show_eval_progress "$eval_done" "$eval_total" "$nframe"
  fi

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
        eval_done=$((eval_done + 1))
        show_eval_progress "$eval_done" "$eval_total" "$nframe"
      fi

      if [[ -f "$metrics_path" ]]; then
        append_record "$records_file" "$model_root_name" "$ckpt_step" "$metrics_path"
        # JJ: live trend refresh while online eval is running; no effect for RUN_EVAL=0 offline replot mode.
        if [[ "$RUN_EVAL" == "1" && "$PLOT_TREND" == "1" && "$LIVE_PLOT_WHILE_EVAL" == "1" ]]; then
          out_prefix="${TREND_DIR}/${MODEL_TYPE}-$(short_model_label "$model_root_name")-nframe${nframe}-${RUN_STAMP}"
          plot_one_model_trend "$model_root_name" "$nframe" "$records_file" "$out_prefix"
          if [[ "$PLOT_COMBINED" == "1" ]]; then
            plot_combined_trend "$nframe" "$records_file" "${TREND_DIR}/combined-${MODELS_TAG}-nframe${nframe}-${RUN_STAMP}"
          fi
        fi
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
      out_prefix="${TREND_DIR}/${MODEL_TYPE}-$(short_model_label "$model_root_name")-nframe${nframe}-${RUN_STAMP}"
      plot_one_model_trend "$model_root_name" "$nframe" "$records_file" "$out_prefix"

      log "Manual single-model replot command:"
      log "python ${PLOT_UTIL_PY} --mode single --records_file ${records_file} --nframe ${nframe} --out_prefix ${out_prefix} --model_root_name ${model_root_name}"
    done

    if [[ "$PLOT_COMBINED" == "1" ]]; then
      plot_combined_trend "$nframe" "$records_file" "${TREND_DIR}/combined-${MODELS_TAG}-nframe${nframe}-${RUN_STAMP}"

      log "Manual combined replot command:"
      log "python ${PLOT_UTIL_PY} --mode combined --records_file ${records_file} --nframe ${nframe} --out_prefix ${TREND_DIR}/combined-${MODELS_TAG}-nframe${nframe}-${RUN_STAMP}"
    fi
  fi

  log "Finished nframe=${nframe}. Trend outputs in: $TREND_DIR"
done

log "All done."
