import os, math
import torch
from transformers import T5Tokenizer
from neuronx_distributed.trace import parallel_model_load
from huggingface_hub import snapshot_download
import time

model_id=os.environ['MODEL_ID']
repo_id=os.environ['COMPILED_MODEL_ID']
local_dir=snapshot_download(repo_id,allow_patterns="tp_*.pt")
max_sequence_length = int(os.environ['MAX_SEQ_LEN'])
t5_tokenizer = T5Tokenizer.from_pretrained(model_id)
t5_tokenizer.model_max_length = max_sequence_length
embedding_t5_model = parallel_model_load(local_dir)


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
            print(f"⏱️   Requests/sec  avg: {avg_rps:.2f},  min: {min(self.rps_list):.2f},  max: {max(self.rps_list):.2f}")
        if self.in_tokens_list:
            avg_in = sum(self.in_tokens_list)/total
            print(f"🔤 Input tokens   avg: {avg_in:.1f},  min: {min(self.in_tokens_list)},  max: {max(self.in_tokens_list)}")
        if self.out_tokens_list:
            avg_out = sum(self.out_tokens_list)/total
            print(f"🔡 Output tokens  avg: {avg_out:.1f},  min: {min(self.out_tokens_list)},  max: {max(self.out_tokens_list)}")
        print(f"🔢 Total executions: {total}")

def get_t5_embedding(text):
    #print(f"Encoding text: {text[:100]}...")
    inputs = t5_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_sequence_length)

    with torch.no_grad():
        output = embedding_t5_model(inputs["input_ids"], inputs["attention_mask"])

    if isinstance(output, dict):
        last_hidden_state = output["last_hidden_state"]  # Extract correct tensor
    else:
        last_hidden_state = output  # Fallback if output isn't a dict (rare case)

    embedding = last_hidden_state.mean(dim=1).squeeze().to(torch.float32).cpu().numpy()
    #print(f"Generated embedding (first 5 dims): {embedding[:5]}")
    return embedding

def warmup_model(model, calls: int = 5,collector=None):
    print(f"🔄 Warming up model with {calls} full passes…")
    for _ in range(calls):
        for pmpt in PROMPTS:
            get_t5_embedding(pmpt)
    print("✅ Warm-up complete.\n")

PROMPTS = ["The image is a close-up photograph of a small, fluffy dog with its tongue out. The dog has light-brown fur and dark eyes. Its black nose is prominent, and its tongue is sticking out of its mouth. The dog appears to be a young puppy, and its fur is shaggy and unkempt. The background is blurred, but it appears to be a green outdoor setting. The overall atmosphere of the image is playful and happy, as the dog's tongue out and happy expression suggest that it is enjoying itself.","The image is a blurry photograph of three people walking through a field on a foggy day. Three people are walking through a field of tall, yellow grass. The people are in the center of the image, and they are blurry and indistinct. There are trees in the background, and the sky is foggy and yellow. The atmosphere is peaceful and serene, with the fog adding a sense of mystery to the scene.",
            "You\'re likely thinking of Rome, the capital of Italy! Rome is famous for many things, including:\n\n1. **Ancient History and Architecture**: Rome is home to numerous ancient ruins, such as the Colosseum, the Roman Forum, and the Pantheon, which showcase the city\'s rich history and engineering prowess.\n2. **Vatican City**: The Vatican, an independent city-state within Rome, is home to the Pope and the central government of the Catholic Church. The Vatican Museums and Sistine Chapel are world-renowned for their art and architecture.\n3. **Food and Wine**: Rome is famous for its delicious cuisine, including dishes like carbonara, amatriciana, and cacio e pepe. The city is also known for its wine production, particularly the Frascati and Castelli Romani wines.\n4. **Art and Culture**: Rome has a vibrant arts scene, with numerous museums, galleries, and festivals throughout the year. The city is also home to the famous Trevi Fountain, Spanish Steps, and Piazza Navona.\n5. **Fashion and Design**: Rome is a hub for fashion and design, with many high-end fashion brands and designers calling the city home.\n6. **History of the Roman Empire**: Rome is the birthplace of the Roman Empire, and the city\'s history is still visible in its architecture, art, and culture.\n7. **Papal History**: Rome has been the center of the Catholic Church for centuries, and the city is home to many important papal landmarks, such as St. Peter\'s Basilica and the Vatican Library.\n8. **Outdoor Spaces**: Rome has many beautiful parks and gardens, such as the Villa Borghese and the Orto Botanico, which offer a peaceful escape from the city\'s bustling streets.\n9. **Nightlife**: Rome has a lively nightlife scene, with many bars, clubs, and live music venues to choose from.\n10. **Film and Media**: Rome has been the setting for many famous films, including La Dolce Vita, Roman Holiday, and Gladiator, and the city is often referred to as the \"Hollywood of Europe.\"\n\nThese are just a few examples of what Rome is famous for. The city has a rich history, culture, and beauty that makes it a must-visit destination for anyone interested in exploring Italy.",
            "The image features a small, tan and white dog standing on a skateboard. The dog's short, fluffy fur is predominantly tan, with a white chest and black nose. Its ears are pointed upright, and its tail is curled up and to the left. The dog's front paws are positioned on the board, while its back paws are raised off the ground. The skateboard has red wheels and a light-colored wooden deck. The background of the image is a plain, off-white color, with a subtle shadow cast by the dog and skateboard. The overall atmosphere of the image is playful and fun, with the dog appearing to be enjoying itself on the skateboard."
          ]

warmup_model(embedding_t5_model,3)

in_tokens_list  = []
out_tokens_list = []
rps_list        = []

latency_collector = LatencyCollector()

for i in range(1,50):
  for pmpt in PROMPTS:
    start_time = time.time()
    outputs=get_t5_embedding(pmpt)
    latency_sec = time.time() - start_time
    rps = 1.0/latency_sec if latency_sec>0 else 0.0
    in_count  = len(t5_tokenizer(pmpt, add_special_tokens=False)["input_ids"])
    #out_text  = outputs[0].text
    #out_count = len(t5_tokenizer(out_text, add_special_tokens=False)["input_ids"])
    latency_collector.record(latency_sec,rps=rps,in_tokens=in_count,out_tokens=in_count)

latency_collector.report(model_id)
