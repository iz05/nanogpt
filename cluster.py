import math
import inspect
from dataclasses import dataclass
from datasets import load_dataset

import os
import pickle
from contextlib import nullcontext
import tiktoken
import pandas as pd
from model import GPT, GPTConfig

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import GridSearchCV

# Parameters
out_dir = 'out-tinystories'  # Directory where the model is saved
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# Set up PyTorch device and context
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load the model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

# Load encoding/decoding
meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
if os.path.exists(meta_path):
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# def find_best_k(data, max_k=10):
#     best_k = 2
#     best_score = -1
#     best_labels = None

#     for k in range(2, max_k + 1):
#         kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#         labels = kmeans.fit_predict(data)
#         score = silhouette_score(data, labels)  # Silhouette Score

#         if score > best_score:
#             best_k = k
#             best_score = score
#             best_labels = labels

#     return best_k, best_labels, best_score

# def cluster(input_string, model):
#     # Encode the input string
#     start_ids = encode(input_string)
#     x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

#     with torch.no_grad():
#         with ctx:
#             outbeddings = model.outbeddings(x)
#             outbedding_shape = outbeddings.shape
#             print("Outbeddings shape:", outbedding_shape)
    
#     token_outbeddings = outbeddings.squeeze(0).cpu().numpy()
#     best_k, best_labels, best_score = find_best_k(token_outbeddings, max_k=10)
#     print(f"Best K: {best_k}, Silhouette Score: {best_score}")

#     num_tokens = token_outbeddings.shape[0]
#     tokens = [decode([start_ids[i]]) for i in range(num_tokens)]

#     df = pd.DataFrame(token_outbeddings)
#     df["token"] = tokens
#     df["label"] = best_labels

#     return df

# Define custom scoring for GridSearchCV
def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    if len(set(labels)) < 2:  # Silhouette needs at least 2 clusters
        return -1
    return silhouette_score(X, labels)

def cluster(X):
    # Model and parameter grid
    param_grid = {
        'n_clusters': [i for i in range(1, int(len(X) ** 0.5))],
        'n_neighbors': [i for i in range(5, 30, 5)],
        'affinity': ['nearest_neighbors']
    }

    # GridSearchCV setup
    model = SpectralClustering(random_state=42, assign_labels='kmeans')
    grid_search = GridSearchCV(model, param_grid, scoring=silhouette_scorer, cv=3)
    grid_search.fit(X)

    # Best parameters and score
    print("Best Params:", grid_search.best_params_)
    print("Best Silhouette Score:", grid_search.best_score_)

    # Plot the best clustering result
    best_model = grid_search.best_estimator_
    best_labels = best_model.fit_predict(X)

    return best_labels

def get_df(input_string, model):
    start_ids = encode(input_string)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        with ctx:
            embeddings = model.get_embeddings(x)
            print("Embeddings shape:", embeddings[0].shape)

    embeddings_squeezed = [embedding.squeeze(0).cpu().tolist() for embedding in embeddings]
    num_tokens = len(embeddings_squeezed[0])
    
    df = pd.DataFrame({
        f"emb{i}" : [emb for emb in embeddings_squeezed[i]] for i in range(len(embeddings_squeezed))
    })

    for idx, embs in enumerate(embeddings_squeezed):
        best_labels = cluster(embs)
        df[f"label{idx}"] = best_labels

    df["token"] = [decode([start_ids[i]]) for i in range(num_tokens)]

    return df

dir_path = 'out-tinystories-cluster-spectral' # where to save the cluster data
os.makedirs(dir_path, exist_ok=True)
print(f"Directory {dir_path} created successfully.")

max_elements = 1000
join_stories = 10
dataset = load_dataset("roneneldan/TinyStories", split="train")
dataset = dataset.select(range(min(len(dataset), max_elements)))
texts = [example["text"] for example in dataset]
training = ["<|endofstory|>".join(texts[i:i+join_stories]) for i in range(0, len(texts) - join_stories, join_stories)]
print(f"Total {len(training)} training.")

for idx, story in enumerate(training):
    print(f"clustering story {idx}...")
    df = get_df(story, model)
    df.to_csv(f"{dir_path}/{idx}.csv", index = False)
    print(f"done with {idx}.\n")

