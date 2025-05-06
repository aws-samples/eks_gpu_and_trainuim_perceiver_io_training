import math
import time
import torch
import os
import sys
import yaml
import requests
from PIL import Image
from vllm import LLM, SamplingParams, TextPrompt
from neuronx_distributed_inference.models.mllama.utils import add_instruct
from huggingface_hub import create_repo,upload_folder,login,snapshot_download
from transformers import AutoTokenizer

hf_token = os.environ['HUGGINGFACE_TOKEN'].strip()
model_id=os.environ['MODEL_ID']
os.environ['NEURON_COMPILED_ARTIFACTS']=model_id
os.environ['VLLM_NEURON_FRAMEWORK']='neuronx-distributed-inference'
login(hf_token,add_to_git_credential=True)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

if len(sys.argv) <= 1:
    print("Error: Please provide a path to a YAML configuration file.")
    sys.exit(1)

config_path = sys.argv[1]
with open(config_path, 'r') as f:
    model_vllm_config_yaml = f.read()

model_vllm_config = yaml.safe_load(model_vllm_config_yaml)

class LatencyCollector:
    def __init__(self):
        self.latency_list = []
        self.rps_list= []
        self.in_tokens_list= []
        self.out_tokens_list= []


    def record(self, latency_sec, rps=None, in_tokens=None, out_tokens=None):
        self.latency_list.append(latency_sec)
        if rps is not None: self.rps_list.append(rps)
        if in_tokens  is not None: self.in_tokens_list.append(in_tokens)
        if out_tokens is not None: self.out_tokens_list.append(out_tokens)


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
        print(f"\n📊 TEST REPORT for {test_name}")
        total = len(self.latency_list)
        for p in [0, 50, 90, 95, 99, 100]:
            value = self.percentile(p) * 1000
            print(f"Latency P{p}: {value:.2f} ms")
        if self.rps_list:
            avg_rps = sum(self.rps_list)/total
            print(f"⏱️  Requests/sec  avg: {avg_rps:.2f},  min: {min(self.rps_list):.2f},  max: {max(self.rps_list):.2f}")
        if self.in_tokens_list:
            avg_in = sum(self.in_tokens_list)/total
            print(f"🔤 Input tokens   avg: {avg_in:.1f},  min: {min(self.in_tokens_list)},  max: {max(self.in_tokens_list)}")
        if self.out_tokens_list:
            avg_out = sum(self.out_tokens_list)/total
            print(f"🔡 Output tokens  avg: {avg_out:.1f},  min: {min(self.out_tokens_list)},  max: {max(self.out_tokens_list)}")
        print(f"🔢 Total executions: {total}")

def get_image(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image

# Model Inputs
PROMPTS = ["What is in this image? Tell me a story",
            "What is the recipe of mayonnaise in two sentences?" ,
            "Describe this image",
            "What is the capital of Italy famous for?",
          ]
IMAGES = [get_image("https://github.com/meta-llama/llama-models/blob/main/models/resources/dog.jpg?raw=true"),
          torch.empty((0,0)),
          get_image("https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/nxd-inference-block-diagram.jpg"),
          torch.empty((0,0)),
         ]
SAMPLING_PARAMS = [dict(top_k=1, temperature=1.0, top_p=1.0, max_tokens=256),
                   dict(top_k=1, temperature=0.9, top_p=1.0, max_tokens=256),
                   dict(top_k=10, temperature=0.9, top_p=0.5, max_tokens=512),
                   dict(top_k=10, temperature=0.75, top_p=0.5, max_tokens=1024),
                  ]


def get_VLLM_mllama_model_inputs(prompt, single_image, sampling_params):
    input_image = single_image
    has_image = torch.tensor([1])
    if isinstance(single_image, torch.Tensor) and single_image.numel() == 0:
        has_image = torch.tensor([0])

    instruct_prompt = add_instruct(prompt, has_image)
    inputs = TextPrompt(prompt=instruct_prompt)
    inputs["multi_modal_data"] = {"image": input_image}
    # Create a sampling params object.
    sampling_params = SamplingParams(**sampling_params)
    return inputs, sampling_params

def warmup_model(model, calls: int = 5,collector=None):
    """
    Run a few dummy inferences over all prompt/image pairs
    to compile kernels and fill caches before measuring.
    """
    print(f"🔄 Warming up model with {calls} full passes…")
    for _ in range(calls):
        for pmpt, img, params in zip(PROMPTS, IMAGES, SAMPLING_PARAMS):
            inp, sp = get_VLLM_mllama_model_inputs(pmpt, img, params)
            _ = model.generate(inp, sp)
    print("✅ Warm-up complete.\n")


def print_outputs(outputs):
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


llm_model = LLM(**model_vllm_config)

assert len(PROMPTS) == len(IMAGES) == len(SAMPLING_PARAMS), \
f"""Text, image prompts and sampling parameters should have the same batch size,
    got {len(PROMPTS)}, {len(IMAGES)}, and {len(SAMPLING_PARAMS)}"""

warmup_model(llm_model, calls=3)
latency_collector = LatencyCollector()
tokenizer= AutoTokenizer.from_pretrained(model_id, use_fast=True)
in_tokens_list  = []
out_tokens_list = []
rps_list        = []

for i in range(1,5):
  for pmpt, img, params in zip(PROMPTS, IMAGES, SAMPLING_PARAMS):
        inputs, sampling_params = get_VLLM_mllama_model_inputs(pmpt, img, params)
        start_time = time.time()
        outputs = llm_model.generate(inputs, sampling_params)
        latency_sec = time.time() - start_time

        rps = 1.0/latency_sec if latency_sec>0 else 0.0
        in_count  = len(tokenizer(pmpt, add_special_tokens=False)["input_ids"])
        if isinstance(img, Image.Image):
            patch_size = 16
            w, h = img.size
            num_patches = (h // patch_size) * (w // patch_size)
            in_count += num_patches
        out_text  = outputs[0].outputs[0].text
        out_count = len(tokenizer(out_text, add_special_tokens=False)["input_ids"])

        latency_collector.record(latency_sec,rps=rps,in_tokens=in_count, out_tokens=out_count)
        print_outputs(outputs)

latency_collector.report(model_id)
