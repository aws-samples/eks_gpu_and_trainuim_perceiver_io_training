# Cold-Start Recommendations with vLLM and AWS Trainium

This repo demonstrates how to solve the **cold-start problem in recommendation systems** using **large language models (LLMs)** on **AWS Trainium (Trn1)** with **vLLM**, **Neuron SDK**, and **FAISS** for semantic retrieval. It features multi-LLM comparison (DeepSeek LLaMA 8B vs. 70B), structured prompting for interest expansion, and high-performance inference using **NeuronX Distributed (NxD)**.

## 🚀 About

End-to-end solution for cold-start recommendations using **vLLM**, **DeepSeek LLaMA (8B & 70B)**, and **FAISS** on **AWS Trainium (Trn1)** with the **Neuron SDK** and **NeuronX Distributed**. Includes LLM-based interest expansion, embedding comparisons (T5 & SentenceTransformers), and scalable retrieval workflows.

## 🛠 Tech Stack

- **Inference**: [vLLM](https://github.com/vllm-project/vllm), DeepSeek LLaMA (8B/70B)
- **Hardware**: AWS EC2 Trn1 (Trainium), Neuron SDK, NeuronX Distributed (NxD)
- **Embeddings**: SentenceTransformers, T5 Encoder
- **Retrieval**: FAISS (Facebook AI Similarity Search)
- **Frameworks**: PyTorch, Hugging Face Transformers
- **Data**: Amazon Books (from the Amazon Reviews dataset)

## 📦 Project Structure

- **notebooks/**
  - **BookExpanSim.ipynb**  
    Uses vLLM to generate expanded user interests from minimal input.
    Converts interests and content into embeddings and builds FAISS indices.
    Retrieves recommendations using FAISS and compares outputs from different LLMs.

- **scripts/**
  - **compile_llm_and_encoders.sh**
    Script to compile `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B`, `meta-llama/Llama-3.1-8B-Instruct` and `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` LLMs using vLLM with NeuronX Distributed on AWS Trainium.
  - **expand_interest_generate_faiss_index.sh**
    Script to expand user interest with FAISS index creation
  - **cell_*.py** 
    Cell scripts that expands user interest and similarity index creation


## ⚙️ Quickstart

1. **Launch a Trn1 instance** with AWS Deep Learning Containers (DLC) and Neuron SDK pre-installed.
2. **Install Python dependencies:**
   ```bash
   pip install --upgrade pip
   pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
   pip install --upgrade neuronx-cc transformers_neuronx neuronx_distributed transformers torch-neuronx accelerate triton protobuf sentence_transformers
   git clone -b v0.6.x-neuron https://github.com/aws-neuron/upstreaming-to-vllm.git
   cd upstreaming-to-vllm
   pip install -r requirements-neuron.txt
   VLLM_TARGET_DEVICE="neuron" && pip install -e .
   pip install --upgrade "transformers==4.45.2"
   ```

   **Clone the Repository**  
   ```bash
   git clone https://github.com/aws-samples/coldstart-recs-on-aws-trainium.git
   cd coldstart-recs-on-aws-trainium
   ```

3. **Run the Jupyter Notebooks**  
   Start Jupyter Notebook to run the interactive examples:
   ```bash
   jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
   ```

4. **Run the model compile script**  
   Run the script:
   ```bash
   compile_llm_and_encoders.sh > compile_llm_and_encoders.log 2>&1 &
   ```

5. **Run the user interest expansion script**
   Edit the `NROWS` to the desired number of rows to include from the Amazon Book Review Dataset
   ```bash
   expand_interest_generate_faiss_index.sh > expand_interest_generate_faiss_index.log 2>&1 &
   ```

6. Then, execute the scrupts and cell in the notebooks in this order:
   - **.BookExpanSim.ipynb**: Retrieve recommendations and compare results from multiple LLMs.
