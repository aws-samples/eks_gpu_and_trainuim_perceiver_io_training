from vllm import LLM, SamplingParams
import yaml
import os
import sys
import torch_neuronx
import pandas as pd
from huggingface_hub import create_repo,upload_folder,login,snapshot_download
from tqdm import tqdm
import time
import math

hf_token = os.environ['HUGGINGFACE_TOKEN'].strip()
books_df_dataset=os.environ['BOOKS_DF_DS']
books_df_dataset_expanded_interest=os.environ['BOOKS_DF_DS_EXP_INTEREST']
repo_id=os.environ['MODEL_ID']
repo_dir=repo_id
os.environ['NEURON_COMPILED_ARTIFACTS']=repo_id
os.environ['VLLM_NEURON_FRAMEWORK']='neuronx-distributed-inference'

login(hf_token,add_to_git_credential=True)


#snapshot_download(repo_id=repo_id,local_dir=repo_dir)
#print(f"Repository '{repo_id}' downloaded to '{repo_dir}'.")

books_df = pd.read_pickle(books_df_dataset)
print(f"Loaded the dataset {books_df_dataset}")

if len(sys.argv) <= 1:
    print("Error: Please provide a path to a YAML configuration file.")
    sys.exit(1)

config_path = sys.argv[1]
with open(config_path, 'r') as f:
    model_vllm_config_yaml = f.read()

class LatencyCollector:
    def __init__(self):
        self.latency_list = []

    def record(self, latency_sec):
        self.latency_list.append(latency_sec)

    def percentile(self, percent):
        if not self.latency_list:
            return 0.0
        latency_list = sorted(self.latency_list)
        pos_float = len(latency_list) * percent / 100
        max_pos = len(latency_list) - 1
        pos_floor = min(math.floor(pos_float), max_pos)
        pos_ceil = min(math.ceil(pos_float), max_pos)
        return latency_list[pos_ceil] if pos_float - pos_floor > 0.5 else latency_list[pos_floor]

    def report(self, test_name="Batch Inference"):
        print(f"\n📊 LATENCY REPORT for {test_name}")
        for p in [0, 50, 90, 95, 99, 100]:
            value = self.percentile(p) * 1000
            print(f"Latency P{p}: {value:.2f} ms")

latency_collector = LatencyCollector()

model_vllm_config = yaml.safe_load(model_vllm_config_yaml)
llm_model = LLM(**model_vllm_config)

def expand_interest(user_reviews):
    """
    Expands the user's book interest using LLM reasoning.
    Ensures the returned type is a valid string for embeddings.
    """
    prompt = f"Expand the following book interests: {user_reviews}"
    sampling_params = SamplingParams(top_k=64)
    request_output = llm_model.generate([prompt],sampling_params=sampling_params)[0]
    # Extract text from request_output
    if request_output.outputs and hasattr(request_output.outputs[0], 'text'):
        expanded_text = request_output.outputs[0].text
    else:
        expanded_text = str(request_output)
    return expanded_text.strip()

# Test the expand_interest function
test_user_interest = "I love epic sci-fi novels with deep world-building."
expanded_test_interest = expand_interest(test_user_interest)
print("✅ Expanded Interest Example:", expanded_test_interest)
# Expand interests for all users
# books_df["expanded_interest"] = books_df["review/text"].apply(expand_interest)

batch_size = 8  # Adjust based on memory availability
MAX_PROMPT_LEN = 512

expanded_interests = []

# Convert reviews to list for slicing
all_reviews = books_df["review/text"].tolist()

# Process in batches
for i in tqdm(range(0, len(all_reviews), batch_size), desc="Expanding interests"):
    batch = all_reviews[i:i + batch_size]
    prompts = [f"Expand the following book interests: {review[:MAX_PROMPT_LEN]}" for review in batch]

    try:
        sampling_params = SamplingParams(top_k=64)
        print(f"⏳ Batch {i}-{i+len(batch)}: sending {len(prompts)} prompts")
        start_time = time.time()
        results = llm_model.generate(prompts, sampling_params=sampling_params)
        latency_sec = time.time() - start_time
        latency_collector.record(latency_sec)
        print(f"✅ Batch {i}-{i+len(batch)} complete in {latency_sec:.2f}s")

        for output in results:
            if output.outputs and hasattr(output.outputs[0], 'text'):
                expanded_text = output.outputs[0].text.strip()
            else:
                expanded_text = str(output)
            expanded_interests.append(expanded_text)

    except Exception as e:
        print(f"⚠️ Error during batch {i}-{i+batch_size}: {e}")
        expanded_interests.extend(["<ERROR: skipped>"] * len(batch))
        with open("skipped_batches.txt", "a") as logf:
            logf.write(f"\n--- Skipped batch {i}-{i+len(batch)} ---\n")
            for review in batch:
                logf.write(review + "\n")
        # maintain DataFrame alignment
        # expanded_interests.extend([""] * len(batch))

# Assign the expanded results to DataFrame
books_df["expanded_interest"] = expanded_interests

books_df.to_pickle(books_df_dataset_expanded_interest)
print(f"✅ Expanded Interests Generated in {books_df_dataset_expanded_interest}!")
latency_collector.report("User Interest Expansion")
