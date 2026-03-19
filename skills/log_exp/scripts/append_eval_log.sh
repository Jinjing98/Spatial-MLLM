#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  bash skills/log_exp/scripts/append_eval_log.sh --exp-id-ref <exp_id_ref> [options]

Options:
  --exp-id-ref <val>   Required. Experiment id reference (e.g. lvsm_01).
  --job-id <val>       Optional. Default: \
auto-detected (slurm job id, or local/tmux run id)
  --status <val>       Optional. Default: submitted
  --csv-path <path>    Optional. Default: experiments/evaluation.csv
  --script-path <path> Optional. Default: scripts/evaluation/evaluate_vsibench_checkpoints_trend.sh
  --force              Optional. Allow duplicate job_id append.
USAGE
}

CSV_PATH="experiments/evaluation.csv"
SCRIPT_PATH="scripts/evaluation/evaluate_vsibench_checkpoints_trend.sh"
EXP_ID_REF=""
JOB_ID=""
STATUS="submitted"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-id-ref)
      EXP_ID_REF="${2:-}"
      shift 2
      ;;
    --job-id)
      JOB_ID="${2:-}"
      shift 2
      ;;
    --status)
      STATUS="${2:-}"
      shift 2
      ;;
    --csv-path)
      CSV_PATH="${2:-}"
      shift 2
      ;;
    --script-path)
      SCRIPT_PATH="${2:-}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$EXP_ID_REF" ]]; then
  echo "[ERROR] --exp-id-ref is required." >&2
  usage
  exit 1
fi

# JJ: support both slurm and local/tmux runs; generate a stable non-empty run id when no slurm job id is available.
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

expected_header="eval_id,exp_id_ref,date,job_id,test_config,status"
actual_header="$(head -n 1 "$CSV_PATH")"
if [[ "$actual_header" != "$expected_header" ]]; then
  echo "[ERROR] CSV header mismatch." >&2
  echo "  expected: $expected_header" >&2
  echo "  actual  : $actual_header" >&2
  exit 1
fi

if [[ "$FORCE" != "1" ]] && rg -n ",${JOB_ID}," "$CSV_PATH" >/dev/null 2>&1; then
  echo "[WARN] job_id=${JOB_ID} already exists in $CSV_PATH. Use --force to append anyway."
  exit 0
fi

read_last_assignment() {
  local var_name="$1"
  local file="$2"
  local line raw
  local -a lines=()
  mapfile -t lines < <(rg -n "^[[:space:]]*${var_name}=" "$file" | cut -d: -f2- || true)
  line=""
  for (( idx=${#lines[@]}-1; idx>=0; idx-- )); do
    raw="${lines[$idx]#*=}"
    # Skip template/self-reference lines in heredoc, e.g. VAR=${VAR}
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

  # Strip one layer of surrounding quotes.
  if [[ "$raw" =~ ^\"(.*)\"$ ]]; then
    raw="${BASH_REMATCH[1]}"
  elif [[ "$raw" =~ ^\'(.*)\'$ ]]; then
    raw="${BASH_REMATCH[1]}"
  fi

  # Resolve ${VAR-default} style to default literal.
  if [[ "$raw" =~ ^\$\{[A-Za-z_][A-Za-z0-9_]*-(.*)\}$ ]]; then
    raw="${BASH_REMATCH[1]}"
  fi

  echo "$raw"
}

as_all_if_empty() {
  local v="$1"
  if [[ -z "${v// }" ]]; then
    echo "ALL"
  else
    echo "$v"
  fi
}

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
if [[ -n "${MODEL_DIRS_CSV// }" ]]; then
  MODEL_SELECTOR="model_dirs=${MODEL_DIRS_CSV}"
fi

TEST_CONFIG="trend.sh | nframe=$(as_all_if_empty "$NFRAMES_LIST") | sampling=$(as_all_if_empty "$SAMPLING") | model_type=$(as_all_if_empty "$MODEL_TYPE") | ${MODEL_SELECTOR} | ckpt_steps=$(as_all_if_empty "$CKPT_STEPS_CSV") | datasets=$(as_all_if_empty "$DATASETS_CSV") | question_types=$(as_all_if_empty "$QUESTION_TYPES_CSV") | scenes=$(as_all_if_empty "$SCENE_NAMES_CSV")"

# Generate eval_id with empty prefix policy: eval_001, eval_002, ...
max_id="$(awk -F, 'NR>1 && $1 ~ /^eval_[0-9]+$/ {gsub(/^eval_/,"",$1); if($1+0>m)m=$1+0} END{print m+0}' "$CSV_PATH")"
next_id=$((max_id + 1))
EVAL_ID="eval_$(printf '%03d' "$next_id")"

TODAY="$(date '+%Y-%m-%d')"
TEST_CONFIG_ESCAPED="${TEST_CONFIG//\"/\"\"}"
NEW_ROW="${EVAL_ID},${EXP_ID_REF},${TODAY},${JOB_ID},\"${TEST_CONFIG_ESCAPED}\",${STATUS}"

printf '%s\n' "$NEW_ROW" >> "$CSV_PATH"
line_no="$(wc -l < "$CSV_PATH")"

echo "[OK] Appended: ${CSV_PATH}"
echo "[OK] Line: ${line_no}"
echo "[OK] Row : ${NEW_ROW}"
