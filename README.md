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

Below is a super-clear, copy-paste version for your README.md's **Project Structure** section:

---

## Project Structure

This repository is organized into several directories. Here’s exactly what each folder/file does:

- **notebooks/**
  - **01_expand_interests.ipynb**  
    Uses vLLM to generate expanded user interests from minimal input.
  - **02_encode_and_index.ipynb**  
    Converts interests and content into embeddings and builds FAISS indices.
  - **03_recommend_and_compare.ipynb**  
    Retrieves recommendations using FAISS and compares outputs from different LLMs.

- **data/**
  - **books_df.pkl**  
    Preprocessed Amazon Books dataset (reviews and metadata).
  - Additional `.pkl` files contain precomputed embeddings and expanded interest outputs.

- **faiss_indices/**
  - Files ending with `_st_faiss.index`  
    FAISS index files built using SentenceTransformer embeddings.
  - Files ending with `_t5_faiss.index`  
    FAISS index files built using T5 encoder embeddings.

- **models/**
  - Contains configuration files and (optional) checkpoints for embedding models (e.g., SentenceTransformers).

- **scripts/**
  - **neuron_inference.py**  
    Script to run vLLM with NeuronX Distributed on AWS Trainium.
  - **benchmark_perf.py**  
    (Optional) Script to benchmark inference performance using NeuronPerf.

- **README.md**  
  This documentation file.

- **requirements.txt**  
  Lists all required Python dependencies for the project.
