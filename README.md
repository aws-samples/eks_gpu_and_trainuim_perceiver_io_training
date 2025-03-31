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

