import os
import torch
import faiss
import pandas as pd
import numpy as np
from transformers import T5Tokenizer
from neuronx_distributed.trace import parallel_model_load
from huggingface_hub import snapshot_download

books_df_dataset_expanded_interest_name=os.environ['BOOKS_DF_DS_EXP_INTEREST']
books_faiss_index=os.environ['BOOKS_DF_FAISS_IDX']

books_df_dataset_expanded_interest = pd.read_pickle(books_df_dataset_expanded_interest_name)

print(f"Loaded dataset path: {os.environ['BOOKS_DF_DS_EXP_INTEREST']}")


model_id=os.environ['MODEL_ID']
repo_id=os.environ['COMPILED_MODEL_ID']
max_sequence_length = int(os.environ['MAX_SEQ_LEN'])
local_dir=snapshot_download(repo_id,allow_patterns="tp_*.pt")

t5_tokenizer = T5Tokenizer.from_pretrained(model_id)
embedding_t5_model = parallel_model_load(local_dir)

def get_t5_embedding(text):
    """
    Create T5-based embeddings by extracting encoder hidden states.
    Ensures inputs are always padded/truncated to the fixed 512 token size.
    """
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


# Generate T5 Embeddings
books_df_dataset_expanded_interest["t5_embedding"] = books_df_dataset_expanded_interest["expanded_interest"].apply(lambda x: get_t5_embedding(x).tolist())

books_df_dataset_expanded_interest.to_pickle(books_df_dataset_expanded_interest_name)
print(f"✅ Updated .pkl with 't5_embedding' column: {books_df_dataset_expanded_interest_name}")

# Create FAISS Index for T5
t5_matrix = np.array(books_df_dataset_expanded_interest["t5_embedding"].tolist()).astype("float32")
faiss.normalize_L2(t5_matrix)
index_t5 = faiss.IndexFlatL2(t5_matrix.shape[1])
index_t5.add(t5_matrix)

# Save to disk
#np.save("t5_matrix.npy", t5_matrix)
faiss.write_index(index_t5,books_faiss_index)
print(f"✅ T5 Embeddings & FAISS index saved in {books_faiss_index}")
xb = np.zeros((5, index_t5.d), dtype=np.float32)
index_t5.reconstruct_n(0, 5, xb)
print("✅ First 5 stored vectors (embeddings):")
print(xb)

