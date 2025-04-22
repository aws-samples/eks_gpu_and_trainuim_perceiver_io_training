import torch
from transformers import T5Tokenizer, T5EncoderModel
from neuronx_distributed.trace import parallel_model_trace, parallel_model_save, parallel_model_load
from pathlib import Path
import torch.multiprocessing as mp
from huggingface_hub import login
import time
import os
import shutil
from huggingface_hub.hf_api import HfFolder
from huggingface_hub import login,snapshot_download,HfApi

hf_token = os.environ['HUGGINGFACE_TOKEN'].strip()
model_id = os.environ['MODEL_ID']
compiled_model_id = os.environ['COMPILED_MODEL_ID']
max_sequence_length = int(os.environ['MAX_SEQ_LEN'])
tp_degree = int(os.environ['TP_DEGREE'])

def forward_wrapper():
    model = T5EncoderModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    return model, {}


if __name__ == '__main__':

    login(hf_token, add_to_git_credential=True)

    mp.set_start_method("spawn", force=True)

    prompt = "This is a test input for compilation."
    compiled_model_path = Path(compiled_model_id)

    model = T5EncoderModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    
    sample_text = "This is a test input for compilation."
    sample_inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        max_length=max_sequence_length,
        truncation=True,
        padding="max_length"
    )
    input_ids = sample_inputs["input_ids"]
    attention_mask = sample_inputs["attention_mask"]

    sample_tensors = (input_ids, attention_mask)
    
    traced_model = parallel_model_trace(forward_wrapper,sample_tensors,tp_degree=tp_degree)
    
    parallel_model_save(traced_model, compiled_model_id)
    print(f"Model compiled successfully! Uploading to {compiled_model_id}")

    api = HfApi()
    api.create_repo(repo_id=compiled_model_id, exist_ok=True)
    api.upload_folder(folder_path=compiled_model_id, repo_id=compiled_model_id, repo_type="model")
