#!/usr/bin/env bash
# Shared helpers for sequential JAX GPU training jobs (WSL2-safe).

detect_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "${PYTHON}"
    return
  fi
  if [[ -x ".venv/bin/python3" ]]; then
    echo ".venv/bin/python3"
    return
  fi
  echo "python3"
}

configure_gpu_env() {
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
}

gpu_cooldown() {
  local sec="${TRAIN_COOLDOWN_SEC:-10}"
  if [[ "${sec}" -gt 0 ]]; then
    echo "GPU cooldown ${sec}s..."
    sleep "${sec}"
  fi
}

is_cuda_failure() {
  local log_file="$1"
  grep -Eiq \
    'CUDA error|CUDA_ERROR|Failed to create stream|Check failed: stream is OK|RESOURCE_EXHAUSTED|Out of memory|core dumped' \
    "${log_file}"
}

run_gpu_train() {
  local label="$1"
  shift
  local max_retries="${TRAIN_GPU_RETRIES:-3}"
  local retry_wait="${TRAIN_GPU_RETRY_WAIT_SEC:-20}"
  local log_file
  log_file="$(mktemp /tmp/gpu_train.XXXXXX.log)"

  for attempt in $(seq 1 "${max_retries}"); do
    echo ">>> ${label} (attempt ${attempt}/${max_retries})"
    set +e
    "$@" 2>&1 | tee "${log_file}"
    local status=${PIPESTATUS[0]}
    set -e
    if [[ "${status}" -eq 0 ]]; then
      rm -f "${log_file}"
      gpu_cooldown
      return 0
    fi
    if is_cuda_failure "${log_file}" && [[ "${attempt}" -lt "${max_retries}" ]]; then
      echo "WARN: GPU/CUDA failure for ${label}; retrying in ${retry_wait}s..."
      sleep "${retry_wait}"
      gpu_cooldown
      continue
    fi
    echo "ERROR: ${label} failed (exit ${status})"
    cat "${log_file}"
    rm -f "${log_file}"
    return "${status}"
  done
}

checkpoint_exists_for_job() {
  local runs_policy="$1"
  local seed="$2"
  local layer_size="${3:-}"
  local py="$4"
  if [[ -n "${layer_size}" ]]; then
    "${py}" -c "
from evaluation.budget_run_discovery import checkpoint_exists
import sys
sys.exit(0 if checkpoint_exists('${runs_policy}', ${seed}, layer_size=${layer_size}) else 1)
"
  else
    "${py}" -c "
from evaluation.budget_run_discovery import checkpoint_exists
import sys
sys.exit(0 if checkpoint_exists('${runs_policy}', ${seed}) else 1)
"
  fi
}

maybe_skip_train() {
  local label="$1"
  local runs_policy="$2"
  local seed="$3"
  local layer_size="${4:-}"
  local py="$5"
  if [[ "${SKIP_IF_CHECKPOINT:-1}" != "1" ]]; then
    return 1
  fi
  if checkpoint_exists_for_job "${runs_policy}" "${seed}" "${layer_size}" "${py}"; then
    echo "=== SKIP ${label} (checkpoint exists for seed ${seed}) ==="
    return 0
  fi
  return 1
}
