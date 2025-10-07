import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

# ---------- basic helpers ----------
def tokenize(text):
    # simple tokenization: words only (keeps dungeness, crab, green_chile as tokens)
    return re.findall(r"\w+", text.lower())

def extract_ingredients(text):
    # tries to parse text like "... ingredients: turmeric, xx, yy instructions: rub ..."
    m = re.search(r"ingredients:(.*?)(instructions:|$)", text.lower(), re.DOTALL)
    if m:
        return m.group(1)
    # fallback: no ingredients: label -> try short heuristic (first portion of text)
    return text[:200]

# a small stoplist (optional)
STOPWORDS = set(["and", "or", "the", "a", "an", "with", "of", "to", "in", "on", "for", "by", "as", "is", "are", "be", "from"])

# ---------- load stuff ----------
df = pd.read_csv("data/searchable_text_for_embeddings.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a standard pre-trained model
dim = model.get_sentence_embedding_dimension()  # Usually 384 for this model

# ---------- embedding functions ----------
def get_embedding(text):
    # Get standard embedding from sentence transformer
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.astype(np.float32)

# ---------- prepare Milvus client and collection ----------
client = MilvusClient("recipes_demo.db")
collection_name = "recipes_collection"

if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

client.create_collection(collection_name=collection_name, dimension=dim)

# ---------- build & insert: one vector per recipe ----------
entities = []
global_id = 0  # unique primary key for each vector row

for _, row in df.iterrows():
    rid = int(row["recipe_id"])
    text = str(row["searchable_text"])

    # Get full text embedding only
    full_emb = get_embedding(text)

    # Insert one row per recipe with full embedding
    entities.append({
        "id": global_id,
        "vector": full_emb.tolist(),
        "text": text,
        "recipe_id": rid,
        "vector_type": "full"
    })
    global_id += 1

# bulk insert
res = client.insert(collection_name=collection_name, data=entities)
print("insert result:", res)

# ---------- searching ----------
def embed_query(query_text):
    return get_embedding(query_text).reshape(1, -1)

query_text = "I have ingredient like tiger prawn ounces. What to make"
q_vec = embed_query(query_text)

res = client.search(
    collection_name=collection_name,
    data=q_vec,
    limit=10,
    output_fields=["text", "recipe_id", "vector_type"]
)
print('\n')
print('\n')
print('\n')
# print top results
for hit in res[0]:
    print("recipe_id:", hit["entity"]["recipe_id"], "vector_type:", hit["entity"]["vector_type"], "distance:", hit["distance"])
    print(hit["entity"]["text"][:200], "...")
    print("----")
