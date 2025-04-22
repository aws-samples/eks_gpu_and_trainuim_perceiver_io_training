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

export BOOKS_DF_DS="books_df.pkl"; export NROWS="100" \
&& python cell_load_books.py

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_1b.pkl" \
&& export BOOKS_DF_DS="books_df.pkl" \
&& export MODEL_ID="yahavb/nxd_vllm_1b" \
&& python cell_expand_interest_llm.py nxd_vllm_1b.yaml > expanded_interest_books_nxd_vllm_1b.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_3b.pkl" \
&& export BOOKS_DF_DS="books_df.pkl" \
&& export MODEL_ID="yahavb/nxd_vllm_3b" \
&& python cell_expand_interest_llm.py nxd_vllm_3b.yaml > expanded_interest_books_nxd_vllm_3b.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_8b.pkl" \
&& export BOOKS_DF_DS="books_df.pkl" \
&& export MODEL_ID="yahavb/nxd_vllm_8b" \
&& python cell_expand_interest_llm.py nxd_vllm_8b.yaml > expanded_interest_books_nxd_vllm_8b.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_70b.pkl" \
&& export BOOKS_DF_DS="books_df.pkl" \
&& export MODEL_ID="yahavb/nxd_vllm_70b" \
&& python cell_expand_interest_llm.py nxd_vllm_70b.yaml > expanded_interest_books_nxd_vllm_70b.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_1b.pkl" \
&& export BOOKS_DF_FAISS_IDX="expanded_interest_books_nxd_vllm_1b_t5_base_faiss.index" \
&& export MODEL_ID="google/t5-v1_1-base" \
&& export COMPILED_MODEL_ID="yahavb/t5-v1_1-base" \
&& export MAX_SEQ_LEN=1024 \
&& python cell_t5_embeddings.py > expanded_interest_books_nxd_vllm_1b_t5_base_faiss.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_1b.pkl" \
&& export BOOKS_DF_FAISS_IDX="expanded_interest_books_nxd_vllm_1b_t5_large_faiss.index" \
&& export MODEL_ID="google/t5-v1_1-base" \
&& export COMPILED_MODEL_ID="yahavb/t5-v1_1-large" \
&& export MAX_SEQ_LEN=1024 \
&& python cell_t5_embeddings.py > expanded_interest_books_nxd_vllm_1b_t5_large_faiss.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_1b.pkl" \
&& export BOOKS_DF_FAISS_IDX="expanded_interest_books_nxd_vllm_1b_t5_xl_faiss.index" \
&& export MODEL_ID="google/t5-v1_1-xl" \
&& export COMPILED_MODEL_ID="yahavb/t5-v1_1-xl" \
&& export MAX_SEQ_LEN=1024 \
&& python cell_t5_embeddings.py > expanded_interest_books_nxd_vllm_1b_t5_xl_faiss.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_3b.pkl" \
&& export BOOKS_DF_FAISS_IDX="expanded_interest_books_nxd_vllm_3b_t5_base_faiss.index" \
&& export MODEL_ID="google/t5-v1_1-base" \
&& export COMPILED_MODEL_ID="yahavb/t5-v1_1-base" \
&& export MAX_SEQ_LEN=1024 \
&& python cell_t5_embeddings.py > expanded_interest_books_nxd_vllm_3b_t5_base_faiss.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_3b.pkl" \
&& export BOOKS_DF_FAISS_IDX="expanded_interest_books_nxd_vllm_3b_t5_large_faiss.index" \
&& export MODEL_ID="google/t5-v1_1-base" \
&& export COMPILED_MODEL_ID="yahavb/t5-v1_1-large" \
&& export MAX_SEQ_LEN=1024 \
&& python cell_t5_embeddings.py > expanded_interest_books_nxd_vllm_3b_t5_large_faiss.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_3b.pkl" \
&& export BOOKS_DF_FAISS_IDX="expanded_interest_books_nxd_vllm_3b_t5_xl_faiss.index" \
&& export MODEL_ID="google/t5-v1_1-xl" \
&& export COMPILED_MODEL_ID="yahavb/t5-v1_1-xl" \
&& export MAX_SEQ_LEN=1024 \
&& python cell_t5_embeddings.py > expanded_interest_books_nxd_vllm_3b_t5_xl_faiss.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_8b.pkl" \
&& export BOOKS_DF_FAISS_IDX="expanded_interest_books_nxd_vllm_8b_t5_base_faiss.index" \
&& export MODEL_ID="google/t5-v1_1-base" \
&& export COMPILED_MODEL_ID="yahavb/t5-v1_1-base" \
&& export MAX_SEQ_LEN=1024 \
&& python cell_t5_embeddings.py > expanded_interest_books_nxd_vllm_8b_t5_base_faiss.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_8b.pkl" \
&& export BOOKS_DF_FAISS_IDX="expanded_interest_books_nxd_vllm_8b_t5_large_faiss.index" \
&& export MODEL_ID="google/t5-v1_1-base" \
&& export COMPILED_MODEL_ID="yahavb/t5-v1_1-large" \
&& export MAX_SEQ_LEN=1024 \
&& python cell_t5_embeddings.py > expanded_interest_books_nxd_vllm_8b_t5_large_faiss.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_8b.pkl" \
&& export BOOKS_DF_FAISS_IDX="expanded_interest_books_nxd_vllm_8b_t5_xl_faiss.index" \
&& export MODEL_ID="google/t5-v1_1-xl" \
&& export COMPILED_MODEL_ID="yahavb/t5-v1_1-xl" \
&& export MAX_SEQ_LEN=1024 \
&& python cell_t5_embeddings.py > expanded_interest_books_nxd_vllm_8b_t5_xl_faiss.log 


export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_70b.pkl" \
&& export BOOKS_DF_FAISS_IDX="expanded_interest_books_nxd_vllm_70b_t5_base_faiss.index" \
&& export MODEL_ID="google/t5-v1_1-base" \
&& export COMPILED_MODEL_ID="yahavb/t5-v1_1-base" \
&& export MAX_SEQ_LEN=1024 \
&& python cell_t5_embeddings.py > expanded_interest_books_nxd_vllm_70b_t5_base_faiss.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_70b.pkl" \
&& export BOOKS_DF_FAISS_IDX="expanded_interest_books_nxd_vllm_70b_t5_large_faiss.index" \
&& export MODEL_ID="google/t5-v1_1-base" \
&& export COMPILED_MODEL_ID="yahavb/t5-v1_1-large" \
&& export MAX_SEQ_LEN=1024 \
&& python cell_t5_embeddings.py > expanded_interest_books_nxd_vllm_70b_t5_large_faiss.log 

export BOOKS_DF_DS_EXP_INTEREST="expanded_interest_books_nxd_vllm_70b.pkl" \
&& export BOOKS_DF_FAISS_IDX="expanded_interest_books_nxd_vllm_70b_t5_xl_faiss.index" \
&& export MODEL_ID="google/t5-v1_1-xl" \
&& export COMPILED_MODEL_ID="yahavb/t5-v1_1-xl" \
&& export MAX_SEQ_LEN=1024 \
&& python cell_t5_embeddings.py > expanded_interest_books_nxd_vllm_70b_t5_xl_faiss.log 
