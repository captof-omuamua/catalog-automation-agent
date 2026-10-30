#!/usr/bin/env bash
# Serve the fine-tuned Qwen3.5-4B tagger on vLLM's OpenAI-compatible API at :8000.
# Same flags as the visual-product-tagging project. Do NOT add
# --structured-outputs-config disable_any_whitespace: the model was fine-tuned on
# '{"value": "X"}' with a space (see reports/parity.md). Answers are capped at 32 tokens,
# so blank-line runaway can't happen.
#   VLLM: path to a vllm binary (any install >= 0.12)   MODEL: model path or Hugging Face id
#   QUANT: set to "fp8" for FP8 weights (Blackwell/Ada/Hopper)   Log: logs/vllm.log
set -euo pipefail
VLLM="${VLLM:-$HOME/catalog-vlm-cascade/.venv-serve/bin/vllm}"
MODEL="${MODEL:-$HOME/catalog-vlm-cascade/outputs/iave-qwen35-4b-merged}"
export VLLM_USE_FLASHINFER_SAMPLER=0  # no CUDA toolkit (nvcc) on this machine
extra=()
[ -n "${QUANT:-}" ] && extra+=(--quantization "$QUANT")
mkdir -p logs
exec "$VLLM" serve "$MODEL" --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization "${GPU_UTIL:-0.88}" \
  --max-num-seqs "${MAX_SEQS:-64}" \
  --limit-mm-per-prompt '{"image": 1}' \
  --default-chat-template-kwargs '{"enable_thinking": false}' "${extra[@]}" > logs/vllm.log 2>&1
