#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SPEC="./meta_graph/meta_graph_3.spec"
DEFAULT_DEVICE_ID="0"
DEFAULT_RUNNER_SCRIPT="$SCRIPT_DIR/musa_run_pb_graph_xla.py"
DEFAULT_WORKDIR="."
DEFAULT_LOG_DIR="log_xla"

usage() {
  cat <<'EOF'
Usage:
  graph_runner_xla.sh --all [options]
  graph_runner_xla.sh --single BS [options]

Options:
  --all                     Run bs=1,2,4,...,4096
  --single BS               Run one batch size
  --spec PATH               Default: ./meta_graph/meta_graph_3.spec
  --pb PATH                 Optional frozen_graph_*.pb path
  --spec-dir DIR            Batch mode over a spec directory
  --device-id ID            MUSA_VISIBLE_DEVICES value. Default: 0
  --runner-script PATH      Default: musa_run_pb_graph_xla.py
  --workdir PATH            Default: current directory
  --log-dir PATH            Default: ./log_xla
  --repeat N                Repeat count per batch size. Default: 5
  --run-iters N             Measured iterations. Default: 20
  --warmup N                Warmup iterations. Default: 3
  --xla                     Enable XLA. Default: enabled
  --no-xla                  Disable XLA
  --xla-dump                Enable HLO dump
  --xla-dump-dir DIR        Override HLO dump directory
  --musa-plugin PATH        Path to libmusa_pjrt_plugin_zy.so
  --average                 Print repeat average. Default: enabled
  --no-average              Disable repeat average output
EOF
}

MODE=""
SINGLE_BS=""
SPEC="$DEFAULT_SPEC"
SPEC_DIR=""
PB=""
DEVICE_ID="$DEFAULT_DEVICE_ID"
RUNNER_SCRIPT="$DEFAULT_RUNNER_SCRIPT"
WORKDIR="$DEFAULT_WORKDIR"
LOG_DIR="$DEFAULT_LOG_DIR"
REPEAT_COUNT="5"
RUN_ITERS="20"
WARMUP="3"
XLA_ENABLED="1"
XLA_DUMP="0"
XLA_DUMP_DIR=""
MUSA_PLUGIN=""
AVERAGE_ENABLED="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      MODE="all"
      shift
      ;;
    --single)
      MODE="single"
      SINGLE_BS="${2:-}"
      if [[ -z "$SINGLE_BS" ]]; then
        echo "error: --single requires a batch size value" >&2
        usage
        exit 1
      fi
      shift 2
      ;;
    --spec)
      SPEC="${2:-}"
      shift 2
      ;;
    --pb)
      PB="${2:-}"
      shift 2
      ;;
    --spec-dir)
      SPEC_DIR="${2:-}"
      shift 2
      ;;
    --device-id)
      DEVICE_ID="${2:-}"
      shift 2
      ;;
    --runner-script)
      RUNNER_SCRIPT="${2:-}"
      shift 2
      ;;
    --workdir)
      WORKDIR="${2:-}"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="${2:-}"
      shift 2
      ;;
    --repeat)
      REPEAT_COUNT="${2:-}"
      shift 2
      ;;
    --run-iters)
      RUN_ITERS="${2:-}"
      shift 2
      ;;
    --warmup)
      WARMUP="${2:-}"
      shift 2
      ;;
    --xla)
      XLA_ENABLED="1"
      shift
      ;;
    --no-xla)
      XLA_ENABLED="0"
      shift
      ;;
    --xla-dump)
      XLA_DUMP="1"
      shift
      ;;
    --xla-dump-dir)
      XLA_DUMP_DIR="${2:-}"
      shift 2
      ;;
    --musa-plugin)
      MUSA_PLUGIN="${2:-}"
      shift 2
      ;;
    --average|--averge)
      AVERAGE_ENABLED="1"
      shift
      ;;
    --no-average|--no-averge)
      AVERAGE_ENABLED="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "error: one of --all or --single is required" >&2
  usage
  exit 1
fi

if ! [[ "$REPEAT_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --repeat must be a positive integer" >&2
  exit 1
fi

if [[ "$MODE" == "single" ]] && ! [[ "$SINGLE_BS" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: batch size must be a positive integer" >&2
  exit 1
fi

if [[ "$MODE" == "all" ]]; then
  BATCH_SIZES=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
else
  BATCH_SIZES=("$SINGLE_BS")
fi

WORKDIR="$(cd "$WORKDIR" && pwd)"
if [[ "$LOG_DIR" = /* ]]; then
  ACTUAL_LOG_DIR="$LOG_DIR"
else
  ACTUAL_LOG_DIR="$WORKDIR/$LOG_DIR"
fi
mkdir -p "$ACTUAL_LOG_DIR"

extract_json_number() {
  local key="$1"
  local log_path="$2"
  grep -oE "['\"]?${key}['\"]?[[:space:]]*[:=][[:space:]]*[0-9]+([.][0-9]+)?" "$log_path" \
    | tail -1 \
    | grep -oE "[0-9]+([.][0-9]+)?$" \
    || true
}

join_by_comma() {
  local out=""
  local item
  for item in "$@"; do
    out+="${out:+, }${item}"
  done
  printf '%s' "$out"
}

compute_average() {
  if [[ $# -eq 0 ]]; then
    printf '%s' ""
    return
  fi
  printf '%s\n' "$@" | awk '
    BEGIN { sum = 0; count = 0; }
    { sum += $1; count += 1; }
    END {
      if (count == 0) { exit 1; }
      printf "%.6f", sum / count;
    }
  '
}

SUMMARY_ROWS=()
FAILED_COUNT=0

for bs in "${BATCH_SIZES[@]}"; do
  log_path="$ACTUAL_LOG_DIR/bs_${bs}.log"
  : > "$log_path"

  bs_status="ok"
  bs_values=()
  bs_trimmed_values=()
  bs_display_values=()
  bs_trimmed_display_values=()
  bs_average=""
  bs_trimmed_average=""

  for ((repeat_idx = 1; repeat_idx <= REPEAT_COUNT; repeat_idx++)); do
    tmp_log="$(mktemp "${TMPDIR:-/tmp}/graph_runner_xla.XXXXXX")"
    status="ok"
    avg_ms=""
    trimmed_avg_ms=""

    runner_args=(--bs "$bs" --run_iters "$RUN_ITERS" --warmup "$WARMUP" --device /device:MUSA:0)
    if [[ -n "$SPEC_DIR" ]]; then
      runner_args+=(--spec_dir "$SPEC_DIR")
    else
      runner_args+=(--spec "$SPEC")
    fi
    if [[ -n "$PB" ]]; then
      runner_args+=(--pb "$PB")
    fi
    if [[ "$XLA_ENABLED" == "1" ]]; then
      runner_args+=(--xla)
    fi
    if [[ "$XLA_DUMP" == "1" ]]; then
      runner_args+=(--xla_dump)
    fi
    if [[ -n "$XLA_DUMP_DIR" ]]; then
      runner_args+=(--xla_dump_dir "$XLA_DUMP_DIR/bs_${bs}_repeat_${repeat_idx}")
    fi
    if [[ -n "$MUSA_PLUGIN" ]]; then
      runner_args+=(--musa_plugin "$MUSA_PLUGIN")
    fi

    if (
      cd "$WORKDIR" && \
      MUSA_ENABLE_TF32=0 \
      MUSA_PINNED_FEED=1 \
      MUSA_PINNED_H2D_ON_COMPUTE_STREAM=1 \
      MUSA_VISIBLE_DEVICES="$DEVICE_ID" \
      python3 "$RUNNER_SCRIPT" "${runner_args[@]}" \
        >"$tmp_log" 2>&1
    ); then
      avg_ms="$(extract_json_number "average_time_ms" "$tmp_log")"
      trimmed_avg_ms="$(extract_json_number "trimmed_avg_ms" "$tmp_log")"
      if [[ -z "$avg_ms" ]]; then
        status="failed"
      fi
      if [[ -z "$trimmed_avg_ms" ]]; then
        trimmed_avg_ms="N/A"
      fi
    else
      status="failed"
      trimmed_avg_ms="N/A"
    fi

    {
      printf '===== bs=%s repeat=%s/%s =====\n' "$bs" "$repeat_idx" "$REPEAT_COUNT"
      cat "$tmp_log"
      printf '\n'
    } >> "$log_path"
    rm -f "$tmp_log"

    if [[ "$status" == "ok" ]]; then
      bs_values+=("$avg_ms")
      bs_display_values+=("$avg_ms")
      if [[ "$trimmed_avg_ms" != "N/A" ]]; then
        bs_trimmed_values+=("$trimmed_avg_ms")
        bs_trimmed_display_values+=("$trimmed_avg_ms")
      else
        bs_trimmed_display_values+=("N/A")
      fi
    else
      FAILED_COUNT=$((FAILED_COUNT + 1))
      bs_status="failed"
      bs_display_values+=("FAILED")
      bs_trimmed_display_values+=("FAILED")
    fi

    echo "bs=${bs} status=${status} average_time_ms={${avg_ms:-FAILED}} trimmed_avg_ms={${trimmed_avg_ms:-FAILED}} log=${log_path}"
  done

  if [[ "$AVERAGE_ENABLED" == "1" ]]; then
    if [[ "${#bs_values[@]}" -gt 0 ]]; then
      bs_average="$(compute_average "${bs_values[@]}")"
      echo "average_repeat={${bs_average}}"
    else
      echo "average_repeat={N/A}"
    fi

    if [[ "${#bs_trimmed_values[@]}" -gt 0 ]]; then
      bs_trimmed_average="$(compute_average "${bs_trimmed_values[@]}")"
      echo "trimmed_avg_repeat={${bs_trimmed_average}}"
    else
      echo "trimmed_avg_repeat={N/A}"
    fi
  fi

  SUMMARY_ROWS+=("${bs}|${bs_status}|$(join_by_comma "${bs_display_values[@]}")|$(join_by_comma "${bs_trimmed_display_values[@]}")|${bs_average}|${bs_trimmed_average}|${log_path}")
  echo
done

echo "Latency Summary"
printf '%s\n' "========================================================================================================================"
printf "%8s  %8s  %24s  %24s  %16s  %18s  %s\n" "bs" "status" "average_time_ms" "trimmed_avg_ms" "average_repeat" "trimmed_avg_repeat" "log"
printf '%s\n' "------------------------------------------------------------------------------------------------------------------------"

FINAL_DATA="{"
SEP=""
for row in "${SUMMARY_ROWS[@]}"; do
  IFS='|' read -r bs status values trimmed_values avg_repeat trimmed_avg_repeat row_log_path <<<"$row"
  avg_repeat_text="${avg_repeat:-N/A}"
  trimmed_avg_repeat_text="${trimmed_avg_repeat:-N/A}"

  printf "%8s  %8s  %24s  %24s  %16s  %18s  %s\n" \
    "$bs" "$status" "{$values}" "{$trimmed_values}" "$avg_repeat_text" "$trimmed_avg_repeat_text" "$row_log_path"

  FINAL_DATA+="${SEP}${bs}: {values=[${values}], trimmed_values=[${trimmed_values}], average_repeat=${avg_repeat_text}, trimmed_avg_repeat=${trimmed_avg_repeat_text}}"
  SEP=", "
done
FINAL_DATA+="}"

printf '%s\n' "------------------------------------------------------------------------------------------------------------------------"
echo "Final performance data:"
echo "$FINAL_DATA"

if [[ "$FAILED_COUNT" -gt 0 ]]; then
  exit 1
fi
