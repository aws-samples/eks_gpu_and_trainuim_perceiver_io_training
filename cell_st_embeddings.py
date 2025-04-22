import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

books_df_dataset_expanded_interest_name=os.environ['BOOKS_DF_DS_EXP_INTEREST']
books_faiss_index=os.environ['BOOKS_DF_FAISS_IDX']

books_df_dataset_expanded_interest = pd.read_pickle(books_df_dataset_expanded_interest_name)

print(f"Loaded dataset path: {os.environ['BOOKS_DF_DS_EXP_INTEREST']}")


# Load SentenceTransformer model
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_st_embedding(text):
    return st_model.encode(text).astype("float32")

# Compute embeddings
books_df_dataset_expanded_interest["st_embedding"] = books_df_dataset_expanded_interest["expanded_interest"].apply(lambda x: get_st_embedding(x).tolist())

books_df_dataset_expanded_interest.to_pickle(books_df_dataset_expanded_interest_name)
print(f"✅ Updated .pkl with 'st_embedding' column: {books_df_dataset_expanded_interest_name}")

# Convert to NumPy matrix
st_matrix = np.array(books_df_dataset_expanded_interest["st_embedding"].tolist()).astype("float32")
#np.save("st_embeddings.npy", st_matrix)  

# Create FAISS index
faiss.normalize_L2(st_matrix)
index_st = faiss.IndexFlatL2(st_matrix.shape[1])
index_st.add(st_matrix)

faiss.write_index(index_st,books_faiss_index)

print(f"✅ SentenceTransformer embeddings computed and saved in {books_faiss_index}")
xb = np.zeros((5, index_st.d), dtype=np.float32)
index_st.reconstruct_n(0, 5, xb)
print("✅ First 5 stored vectors (embeddings):")
print(xb)
