#!/bin/bash
# Serve a single vLLM instance for the SpeechCheck pipeline on a 2-GPU machine:
#   - Port 8901: Qwen3-Omni-30B-A3B-Instruct -> all 3 stages (speech, rules, eval)
#
# Uses both GPUs (0,1) with tensor-parallel-size 2, dtype bfloat16,
# max-model-len 32768. The 30B model (bf16 ~60GB) does not fit on a single GPU,
# so a single TP=2 instance over both GPUs is used instead of two instances.
#
# Run:  bash serve_instruct.sh
# Stop: Ctrl-C (kills the instance) or `pkill -f vllm`.

set -euo pipefail

INSTRUCT_PATH="/home/voice/data/voice/voice-models/Qwen3-Omni-30B-A3B-Instruct/"

VLLM_COMMON_ARGS=(
    --dtype bfloat16
    --max-model-len 32768
    --allowed-local-media-path /
    --trust-remote-code
    --gpu-memory-utilization 0.95
    --enable-chunked-prefill
    --limit-mm-per-prompt '{"image": 3, "video": 3, "audio": 3}'
)

# Ensure the process is killed on Ctrl-C / script exit.
PIDS=()
cleanup() {
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "Stopping vLLM instance (pids: ${PIDS[*]})..."
        kill "${PIDS[@]}" 2>/dev/null || true
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# All stages: Instruct on port 8901, GPUs 0,1
echo "Starting Qwen3-Omni-30B-A3B-Instruct on 127.0.0.1:8901 (GPUs 0,1)..."
CUDA_VISIBLE_DEVICES=0,1 vllm serve "${INSTRUCT_PATH}" \
    --host 127.0.0.1 \
    --port 8901 \
    --tensor-parallel-size 2 \
    "${VLLM_COMMON_ARGS[@]}" &
PIDS+=($!)

echo "vLLM instance launched. Waiting..."
wait
