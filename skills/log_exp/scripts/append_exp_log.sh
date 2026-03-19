#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  bash skills/log_exp/scripts/append_exp_log.sh --exp-id-ref <id> [options]

Options:
  --exp-id-ref <val>   Required. Experiment id reference.
  --mode <val>         Optional. auto|eval|train (default: auto)
  --job-id <val>       Optional. Auto-detected if omitted.
  --status <val>       Optional. Default: submitted
  --csv-path <path>    Optional. Default by mode:
                       eval -> experiments/evaluation.csv
                       train -> experiments/training.csv
  --script-path <path> Optional. Default by mode:
                       eval -> scripts/evaluation/evaluate_vsibench_checkpoints_trend.sh
                       train -> scripts/training/spatial_mllm_train_demo_DDP_LVSM_overfitting_hpc.sh
  --force              Optional. Allow duplicate job_id append.
USAGE
}

MODE="auto"
EXP_ID_REF=""
JOB_ID=""
STATUS="submitted"
CSV_PATH=""
SCRIPT_PATH=""
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-id-ref) EXP_ID_REF="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --job-id) JOB_ID="${2:-}"; shift 2 ;;
    --status) STATUS="${2:-}"; shift 2 ;;
    --csv-path) CSV_PATH="${2:-}"; shift 2 ;;
    --script-path) SCRIPT_PATH="${2:-}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$EXP_ID_REF" ]]; then
  echo "[ERROR] --exp-id-ref is required." >&2
  exit 1
fi

if [[ "$MODE" != "auto" && "$MODE" != "eval" && "$MODE" != "train" ]]; then
  echo "[ERROR] --mode must be one of: auto|eval|train" >&2
  exit 1
fi

# Resolve mode by hints when auto.
if [[ "$MODE" == "auto" ]]; then
  if [[ -n "$SCRIPT_PATH" ]]; then
    if [[ "$SCRIPT_PATH" == *"/evaluation/"* ]]; then MODE="eval"; fi
    if [[ "$SCRIPT_PATH" == *"/training/"* ]]; then MODE="train"; fi
  fi
  if [[ "$MODE" == "auto" && -n "$CSV_PATH" ]]; then
    if [[ "$CSV_PATH" == *"evaluation.csv" ]]; then MODE="eval"; fi
    if [[ "$CSV_PATH" == *"training.csv" ]]; then MODE="train"; fi
  fi
  if [[ "$MODE" == "auto" ]]; then
    echo "[ERROR] Cannot infer mode (eval/train). Please pass --mode eval or --mode train." >&2
    exit 1
  fi
fi

if [[ -z "$CSV_PATH" ]]; then
  if [[ "$MODE" == "eval" ]]; then
    CSV_PATH="experiments/evaluation.csv"
  else
    CSV_PATH="experiments/training.csv"
  fi
fi

if [[ -z "$SCRIPT_PATH" ]]; then
  if [[ "$MODE" == "eval" ]]; then
    SCRIPT_PATH="scripts/evaluation/evaluate_vsibench_checkpoints_trend.sh"
  else
    SCRIPT_PATH="scripts/training/spatial_mllm_train_demo_DDP_LVSM_overfitting_hpc.sh"
  fi
fi

if [[ -z "${JOB_ID// }" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    JOB_ID="${SLURM_JOB_ID}"
  else
    ts="$(date '+%Y%m%d_%H%M%S')"
    host="$(hostname -s 2>/dev/null || echo local)"
    if [[ -n "${TMUX:-}" ]]; then
      JOB_ID="tmux_${host}_${ts}"
    else
      JOB_ID="local_${host}_${ts}"
    fi
  fi
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "[ERROR] CSV not found: $CSV_PATH" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[ERROR] Script not found: $SCRIPT_PATH" >&2
  exit 1
fi

header="$(head -n 1 "$CSV_PATH")"
if [[ "$MODE" == "eval" && "$header" != "eval_id,exp_id_ref,date,job_id,test_config,status" ]]; then
  echo "[ERROR] CSV header does not match eval schema: $CSV_PATH" >&2
  exit 1
fi
if [[ "$MODE" == "train" && "$header" != "exp_id,date,job_id,config,gpus,checkpoint,status" ]]; then
  echo "[ERROR] CSV header does not match train schema: $CSV_PATH" >&2
  exit 1
fi

if [[ "$FORCE" != "1" ]] && rg -n ",${JOB_ID}," "$CSV_PATH" >/dev/null 2>&1; then
  echo "[WARN] job_id=${JOB_ID} already exists in $CSV_PATH. Use --force to append anyway."
  exit 0
fi

read_last_assignment() {
  local var_name="$1"; local file="$2"; local line raw
  local -a lines=()
  mapfile -t lines < <(rg -n "^[[:space:]]*${var_name}=" "$file" | cut -d: -f2- || true)
  line=""
  for (( idx=${#lines[@]}-1; idx>=0; idx-- )); do
    raw="${lines[$idx]#*=}"
    if [[ "$raw" =~ \$\{${var_name}\} ]]; then
      continue
    fi
    line="${lines[$idx]}"
    break
  done
  if [[ -z "$line" && ${#lines[@]} -gt 0 ]]; then
    line="${lines[$((${#lines[@]} - 1))]}"
  fi
  raw="${line#*=}"
  raw="$(echo "$raw" | sed -E 's/[[:space:]]+#.*$//')"
  if [[ "$raw" =~ ^\"(.*)\"$ ]]; then raw="${BASH_REMATCH[1]}"; fi
  if [[ "$raw" =~ ^\'(.*)\'$ ]]; then raw="${BASH_REMATCH[1]}"; fi
  if [[ "$raw" =~ ^\$\{[A-Za-z_][A-Za-z0-9_]*-(.*)\}$ ]]; then raw="${BASH_REMATCH[1]}"; fi
  echo "$raw"
}

resolve_script_ref() {
  local value="$1"
  local file="$2"
  if [[ "$value" =~ ^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$ ]]; then
    local ref_var="${BASH_REMATCH[1]}"
    local resolved
    resolved="$(read_last_assignment "$ref_var" "$file")"
    if [[ -n "${resolved// }" ]]; then
      echo "$resolved"
      return 0
    fi
  fi
  echo "$value"
}

as_all_if_empty() { local v="$1"; [[ -z "${v// }" ]] && echo "ALL" || echo "$v"; }

TODAY="$(date '+%Y-%m-%d')"

if [[ "$MODE" == "eval" ]]; then
  MODEL_TYPE="$(read_last_assignment MODEL_TYPE "$SCRIPT_PATH")"
  MODEL_SEARCH_PATTERN="$(read_last_assignment MODEL_SEARCH_PATTERN "$SCRIPT_PATH")"
  MODEL_DIRS_CSV="$(read_last_assignment MODEL_DIRS_CSV "$SCRIPT_PATH")"
  CKPT_STEPS_CSV="$(read_last_assignment CKPT_STEPS_CSV "$SCRIPT_PATH")"
  DATASETS_CSV="$(read_last_assignment DATASETS_CSV "$SCRIPT_PATH")"
  QUESTION_TYPES_CSV="$(read_last_assignment QUESTION_TYPES_CSV "$SCRIPT_PATH")"
  SCENE_NAMES_CSV="$(read_last_assignment SCENE_NAMES_CSV "$SCRIPT_PATH")"
  NFRAMES_LIST="$(read_last_assignment NFRAMES_LIST "$SCRIPT_PATH")"
  SAMPLING="$(read_last_assignment SAMPLING "$SCRIPT_PATH")"

  MODEL_SELECTOR="model_search=$(as_all_if_empty "$MODEL_SEARCH_PATTERN")"
  if [[ -n "${MODEL_DIRS_CSV// }" ]]; then MODEL_SELECTOR="model_dirs=${MODEL_DIRS_CSV}"; fi

  TEST_CONFIG="trend.sh | nframe=$(as_all_if_empty "$NFRAMES_LIST") | sampling=$(as_all_if_empty "$SAMPLING") | model_type=$(as_all_if_empty "$MODEL_TYPE") | ${MODEL_SELECTOR} | ckpt_steps=$(as_all_if_empty "$CKPT_STEPS_CSV") | datasets=$(as_all_if_empty "$DATASETS_CSV") | question_types=$(as_all_if_empty "$QUESTION_TYPES_CSV") | scenes=$(as_all_if_empty "$SCENE_NAMES_CSV")"

  max_id="$(awk -F, 'NR>1 && $1 ~ /^eval_[0-9]+$/ {gsub(/^eval_/,"",$1); if($1+0>m)m=$1+0} END{print m+0}' "$CSV_PATH")"
  next_id=$((max_id + 1))
  eval_id="eval_$(printf '%03d' "$next_id")"
  esc="${TEST_CONFIG//\"/\"\"}"
  row="${eval_id},${EXP_ID_REF},${TODAY},${JOB_ID},\"${esc}\",${STATUS}"
else
  N_GPU="$(read_last_assignment N_GPU "$SCRIPT_PATH")"
  [[ -z "${N_GPU// }" ]] && N_GPU="1"
  MODEL_TYPE="$(read_last_assignment MODEL_TYPE "$SCRIPT_PATH")"
  RUN_NAME_APPENDIX="$(read_last_assignment RUN_NAME_APPENDIX "$SCRIPT_PATH")"
  LR="$(read_last_assignment lr "$SCRIPT_PATH")"
  LR="$(resolve_script_ref "$LR" "$SCRIPT_PATH")"
  ADAPTOR_LR="$(read_last_assignment LVSM_ADAPTOR_LR "$SCRIPT_PATH")"
  OVERFIT_MODE="$(read_last_assignment OVERFIT_MODE "$SCRIPT_PATH")"
  OVERFIT_MAX="$(read_last_assignment OVERFIT_MAX_TRAIN_SAMPLES "$SCRIPT_PATH")"
  DATASETS="$(read_last_assignment DATASETS "$SCRIPT_PATH")"
  OUTPUT_ROOT="$(read_last_assignment OUTPUT_ROOT "$SCRIPT_PATH")"

  checkpoint=""
  if [[ -n "${OUTPUT_ROOT// }" && -d "$OUTPUT_ROOT" ]]; then
    checkpoint="$(ls -1dt "$OUTPUT_ROOT"/* 2>/dev/null | head -n 1 || true)"
  fi
  [[ -z "$checkpoint" ]] && checkpoint="N/A"

  config="$(as_all_if_empty "$DATASETS") | MODEL_TYPE=$(as_all_if_empty "$MODEL_TYPE") | RUN_NAME_APPENDIX=$(as_all_if_empty "$RUN_NAME_APPENDIX") | lr=$(as_all_if_empty "$LR") | adaptor_lr=$(as_all_if_empty "$ADAPTOR_LR") | overfit=$(as_all_if_empty "$OVERFIT_MODE"):${OVERFIT_MAX:-N/A}"
  row="${EXP_ID_REF},${TODAY},${JOB_ID},${config},${N_GPU},${checkpoint},${STATUS}"
fi

printf '%s\n' "$row" >> "$CSV_PATH"
line_no="$(wc -l < "$CSV_PATH")"
echo "[OK] mode=${MODE}"
echo "[OK] Appended: ${CSV_PATH}"
echo "[OK] Line: ${line_no}"
echo "[OK] Row : ${row}"
