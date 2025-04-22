from vllm import LLM, SamplingParams
import yaml
import os
import sys
from huggingface_hub import create_repo,upload_folder,login

hf_token = os.environ['HUGGINGFACE_TOKEN'].strip()
compiled_model_id=os.environ['COMPILED_MODEL_ID']
os.environ['NEURON_COMPILED_ARTIFACTS']=compiled_model_id
os.environ['VLLM_NEURON_FRAMEWORK']='neuronx-distributed-inference'

if len(sys.argv) <= 1:
  print("Error: Please provide a path to a vLLM YAML configuration file.")
  sys.exit(1)

config_path = sys.argv[1]
with open(config_path, 'r') as f:
  model_vllm_config_yaml = f.read()

login(hf_token,add_to_git_credential=True)

def push_compiled_model_to_hf(local_dir,repo_id,commit_message):
  create_repo(repo_id=repo_id,exist_ok=True,private=False)
  upload_folder(folder_path=local_dir,path_in_repo="",repo_id=repo_id,commit_message=commit_message)

model_vllm_config = yaml.safe_load(model_vllm_config_yaml)
llm_model = LLM(**model_vllm_config)
sampling_params = SamplingParams(top_k=1, temperature=1.0, max_tokens=64)
prompt = "What is Annapurna Labs?"
print(f"Running inference with prompt: '{prompt}'")
outputs = llm_model.generate([prompt], sampling_params)
for output in outputs:
  print("Prompt:", output.prompt)
  print("Generated text:", output.outputs[0].text)

push_compiled_model_to_hf(
  local_dir=compiled_model_id,
  repo_id=compiled_model_id,
  commit_message=f"Add NxD compiled model {compiled_model_id} for vLLM")

print(f"✅  compilation was successful and stored in {compiled_model_id}!")
