# =========================
#  Load Dataset
# =========================
import pandas as pd
import os
import kagglehub
import shutil

books_df_dataset=os.environ['BOOKS_DF_DS']
nrows=os.environ['NROWS']

local_dir="./data"
local_file=os.path.join(local_dir,"Books_rating.csv")
if os.path.exists(local_file):
    print("Books_rating.csv already exists locally. Skipping download.")
else:
  dataset_path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")
  print("Path to dataset files:",dataset_path)
  kaggle_file=os.path.join(dataset_path,"Books_rating.csv")
  os.makedirs(local_dir,exist_ok=True)
  shutil.copy(kaggle_file,local_file)
  print(f"Copied Books_rating.csv to {local_file}")

df = pd.read_csv(local_file,nrows=int(nrows))
books_df = df[['Id', 'Title', 'User_id', 'review/helpfulness', 'review/score', 'review/time', 'review/summary', 'review/text']]
books_df.to_pickle(books_df_dataset)  
print(f"✅ Books dataset loaded & saved in {books_df_dataset}.")
