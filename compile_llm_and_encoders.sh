#!/bin/sh
set -eu

if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
  echo "HUGGINGFACE_TOKEN is not set."
  read -p "Please enter your Hugging Face token: " HUGGINGFACE_TOKEN
  export HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN
  echo
  if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: no Hugging Face token provided. Exiting." >&2
    exit 1
  fi
fi

cat nxd_vllm_1b.yaml \
&& export COMPILED_MODEL_ID="yahavb/nxd_vllm_1b" \
&& python cell_compile_vllm.py nxd_vllm_1b.yaml > nxd_vllm_1b.log

cat nxd_vllm_3b.yaml \
&& export COMPILED_MODEL_ID="yahavb/nxd_vllm_3b" \
&& python cell_compile_vllm.py nxd_vllm_3b.yaml > nxd_vllm_3b.log

cat nxd_vllm_8b.yaml \
&& export COMPILED_MODEL_ID="yahavb/nxd_vllm_8b" \
&& python cell_compile_vllm.py nxd_vllm_8b.yaml > nxd_vllm_8b.log

cat nxd_vllm_70b.yaml \
&& export COMPILED_MODEL_ID="yahavb/nxd_vllm_70b" \
&& python cell_compile_vllm.py nxd_vllm_70b.yaml > nxd_vllm_70b.log

export COMPILED_MODEL_ID="yahavb/t5-v1_1-base" && export MODEL_ID="google/t5-v1_1-base" && export MAX_SEQ_LEN=1024 && export TP_DEGREE=8 && python cell_compile_t5.py

export COMPILED_MODEL_ID="yahavb/t5-v1_1-large" && export MODEL_ID="google/t5-v1_1-large" && export MAX_SEQ_LEN=1024 && export TP_DEGREE=8 && python cell_compile_t5.py

export COMPILED_MODEL_ID="yahavb/t5-v1_1-xl" && export MODEL_ID="google/t5-v1_1-xl" && export MAX_SEQ_LEN=1024 && export TP_DEGREE=8 && python cell_compile_t5.py
