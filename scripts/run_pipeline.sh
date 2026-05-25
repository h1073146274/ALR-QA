#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  source "${ROOT}/.env"
  set +a
fi

ARCH="${ARCH:-qwen}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"

require_var() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "${value}" ]]; then
    echo "missing ${name}" >&2
    exit 1
  fi
}

split_chunks() {
  require_var DOCUMENT_JSONL
  require_var SPLIT_OUT_DIR
  require_var CORPUS_NAME
  require_var TOKENIZER_NAME_OR_PATH
  require_var MIN_CHUNK_TOKENS
  require_var MAX_CHUNK_TOKENS
  require_var CHUNK_OVERLAP_TOKENS
  require_var SAMPLE_DOCS
  require_var SPLIT_SEED
  "${PYTHON_BIN}" "${ROOT}/scripts/split_chunks.py" \
    --input "${DOCUMENT_JSONL}" \
    --out_dir "${SPLIT_OUT_DIR}" \
    --name "${CORPUS_NAME}" \
    --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}" \
    --min_tokens "${MIN_CHUNK_TOKENS}" \
    --max_tokens "${MAX_CHUNK_TOKENS}" \
    --overlap_tokens "${CHUNK_OVERLAP_TOKENS}" \
    --sample_docs "${SAMPLE_DOCS}" \
    --seed "${SPLIT_SEED}"
}

build_dataset() {
  require_var TRAIN_CHUNKS_JSONL
  require_var DATASET_OUT_DIR
  require_var API_KEY
  require_var TEACHER_MODEL
  "${PYTHON_BIN}" "${ROOT}/scripts/build_view_conditioned_dataset.py" \
    --input_jsonl "${TRAIN_CHUNKS_JSONL}" \
    --out_dir "${DATASET_OUT_DIR}" \
    --api_key "${API_KEY}" \
    --base_url "${API_BASE_URL:-}" \
    --model "${TEACHER_MODEL}" \
    --k "${TOP_K_VIEWS:-3}" \
    --max_chunks "${MAX_CHUNKS:-0}" \
    --temperature "${TEACHER_TEMPERATURE:-0.2}"
}

precompute_train_anchors() {
  require_var DATASET_OUT_DIR
  require_var ANCHOR_OUT_DIR
  require_var ENCODER_MODEL
  "${PYTHON_BIN}" "${ROOT}/scripts/precompute_anchors.py" \
    --in_jsonl "${DATASET_OUT_DIR}/view_conditioned_sft.jsonl" \
    --out_dir "${ANCHOR_OUT_DIR}" \
    --encoder_name_or_path "${ENCODER_MODEL}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE:-64}" \
    --max_length "${ANCHOR_MAX_LENGTH:-256}" \
    --bf16 \
    --normalize \
    --encode_view \
    --write_indexed_jsonl \
    --append_keywords
}

train_model() {
  require_var BASE_MODEL
  require_var OUTPUT_DIR
  require_var DOC_ANCHOR
  require_var CHUNK_ANCHOR
  local train_jsonl="${TRAIN_JSONL:-}"
  if [[ -z "${train_jsonl}" ]]; then
    require_var ANCHOR_OUT_DIR
    train_jsonl="${ANCHOR_OUT_DIR}/view_conditioned_sft.with_anchor_index.jsonl"
  fi
  TRAIN_ENTRY="${ROOT}/train/qwen/train.py"
  if [[ "${ARCH}" == "phi" ]]; then
    TRAIN_ENTRY="${ROOT}/train/phi/train.py"
  fi
  local learning_rate="${LR:-}"
  if [[ -z "${learning_rate}" ]]; then
    if [[ "${ARCH}" == "phi" ]]; then
      learning_rate="5e-5"
    else
      learning_rate="1e-4"
    fi
  fi
  "${PYTHON_BIN}" "${TRAIN_ENTRY}" \
    --model_name_or_path "${BASE_MODEL}" \
    --train_jsonl "${train_jsonl}" \
    --doc_anchor_pt "${DOC_ANCHOR}" \
    --chunk_anchor_pt "${CHUNK_ANCHOR}" \
    --view_anchor_pt "${VIEW_ANCHOR:-}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_seq_len "${MAX_SEQ_LEN:-2048}" \
    --apt_token "<|APT|>" \
    --per_device_train_batch_size "${BATCH:-1}" \
    --gradient_accumulation_steps "${GAS:-8}" \
    --num_train_epochs "${EPOCHS:-3}" \
    --max_steps "${MAX_STEPS:--1}" \
    --learning_rate "${learning_rate}" \
    --alpha "${ALPHA:-0.08}" \
    --beta_route "${BETA_ROUTE:-0.05}" \
    --delta_vgc "${DELTA_VGC:-0.05}" \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --save_steps "${SAVE_STEPS:-500}" \
    --seed "${SEED:-42}"
}

precompute_infer_anchors() {
  require_var INPUT_JSONL
  require_var INFER_ANCHOR_OUT_DIR
  require_var ENCODER_MODEL
  "${PYTHON_BIN}" "${ROOT}/scripts/precompute_anchors.py" \
    --in_jsonl "${INPUT_JSONL}" \
    --out_dir "${INFER_ANCHOR_OUT_DIR}" \
    --encoder_name_or_path "${ENCODER_MODEL}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE:-64}" \
    --max_length "${ANCHOR_MAX_LENGTH:-256}" \
    --bf16 \
    --normalize \
    --write_indexed_jsonl \
    --append_keywords
}

infer_model() {
  require_var BASE_MODEL
  require_var LORA_DIR
  require_var INPUT_JSONL
  require_var OUTPUT_JSONL
  require_var DOC_ANCHOR
  require_var CHUNK_ANCHOR
  INFER_ENTRY="${ROOT}/inference/qwen/infer.py"
  if [[ "${ARCH}" == "phi" ]]; then
    INFER_ENTRY="${ROOT}/inference/phi/infer.py"
  fi
  "${PYTHON_BIN}" "${INFER_ENTRY}" \
    --base_model "${BASE_MODEL}" \
    --lora_dir "${LORA_DIR}" \
    --doc_anchor_pt "${DOC_ANCHOR}" \
    --chunk_anchor_pt "${CHUNK_ANCHOR}" \
    --anchor_row_key "${ANCHOR_ROW_KEY:-anchor_row_idx}" \
    --view_prototypes_pt "${VIEW_PROTOTYPES:-}" \
    --input_jsonl "${INPUT_JSONL}" \
    --output_jsonl "${OUTPUT_JSONL}" \
    --log_jsonl "${LOG_JSONL:-}" \
    --trace_jsonl "${TRACE_JSONL:-}" \
    --counts_json "${COUNTS_JSON:-}" \
    --summary_json "${SUMMARY_JSON:-}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --K "${K:-3}" \
    --seed "${SEED:-42}"
}

case "${1:-}" in
  split) split_chunks ;;
  dataset) build_dataset ;;
  anchors) precompute_train_anchors ;;
  train) train_model ;;
  infer-anchors) precompute_infer_anchors ;;
  infer) infer_model ;;
  all)
    split_chunks
    build_dataset
    precompute_train_anchors
    DOC_ANCHOR="${ANCHOR_OUT_DIR}/doc_anchor_emb.pt"
    CHUNK_ANCHOR="${ANCHOR_OUT_DIR}/chunk_anchor_emb.pt"
    VIEW_ANCHOR="${ANCHOR_OUT_DIR}/view_anchor_emb.pt"
    train_model
    ;;
  *)
    echo "usage: scripts/run_pipeline.sh split|dataset|anchors|train|infer-anchors|infer|all" >&2
    exit 1
    ;;
esac
