import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from gensim.models import Word2Vec
from pymilvus import MilvusClient

# ---------- basic helpers ----------
def tokenize(text):
    # simple tokenization: words only (keeps dungeness, crab, green_chile as tokens)
    return re.findall(r"\w+", text.lower())

def extract_ingredients(text):
    # tries to parse text like "... ingredients: turmeric, xx, yy instructions: rub ..."
    m = re.search(r"ingredients:(.*?)(instructions:|$)", text.lower(), re.DOTALL)
    if m:
        return tokenize(m.group(1))
    # fallback: no ingredients: label -> try short heuristic (first 40 words)
    return tokenize(text)[:40]

# a small stoplist (optional)
STOPWORDS = set(["and", "or", "the", "a", "an", "with", "of", "to", "in", "on", "for", "by", "as", "is", "are", "be", "from"])

# ---------- load stuff ----------
df = pd.read_csv("data/searchable_text_for_embeddings.csv")
model = Word2Vec.load("models/Food2Vec.bin")
dim = model.vector_size  # 100

# ---------- build IDF over tokens present in model.wv ----------
N = len(df)
df_counter = Counter()
for txt in df["searchable_text"].astype(str):
    toks = set([t for t in tokenize(txt) if t in model.wv])
    for t in toks:
        df_counter[t] += 1

idf = {}
for t, cnt in df_counter.items():
    idf[t] = math.log((N + 1) / (cnt + 1)) + 1.0  # +1 smoothing

def weighted_avg_embedding(tokens, idf_map, default_idf=1.0):
    vecs = []
    weights = []
    for t in tokens:
        if t in model.wv:
            w = idf_map.get(t, default_idf)
            vecs.append(model.wv[t] * w)
            weights.append(w)
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    emb = np.sum(vecs, axis=0) / (sum(weights) + 1e-9)
    # normalize for cosine
    norm = np.linalg.norm(emb) + 1e-9
    emb = (emb / norm).astype(np.float32)
    return emb

# ---------- prepare Milvus client and collection ----------
client = MilvusClient("recipes_demo.db")
collection_name = "recipes_collection"

if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

client.create_collection(collection_name=collection_name, dimension=dim)

# ---------- build & insert: two vectors per recipe ----------
entities = []
global_id = 0  # unique primary key for each vector row

for _, row in df.iterrows():
    rid = int(row["recipe_id"])
    text = str(row["searchable_text"])

    # tokens
    ing_tokens = [t for t in extract_ingredients(text) if t not in STOPWORDS]
    full_tokens = [t for t in tokenize(text) if t not in STOPWORDS]

    # embeddings
    ing_emb = weighted_avg_embedding(ing_tokens, idf)
    full_emb = weighted_avg_embedding(full_tokens, idf)

    # if you want ingredients to weigh much more in the final 'combined' vector:
    ingredient_weight = 3.0
    combined_emb = (ingredient_weight * ing_emb + full_emb) / (ingredient_weight + 1.0)
    combined_emb /= (np.linalg.norm(combined_emb) + 1e-9)  # final normalize

    # Insert three rows per recipe (optional): ingredients-only, full-only, combined
    entities.append({
        "id": global_id,
        "vector": ing_emb.tolist(),
        "text": text,
        "recipe_id": rid,
        "vector_type": "ingredients"
    })
    global_id += 1

    entities.append({
        "id": global_id,
        "vector": full_emb.tolist(),
        "text": text,
        "recipe_id": rid,
        "vector_type": "full"
    })
    global_id += 1

    entities.append({
        "id": global_id,
        "vector": combined_emb.tolist(),
        "text": text,
        "recipe_id": rid,
        "vector_type": "combined"
    })
    global_id += 1

# bulk insert
res = client.insert(collection_name=collection_name, data=entities)
print("insert result:", res)

# ---------- searching: prefer ingredients vectors ----------
def embed_query_as_ingredients(q):
    q_tokens = [t for t in tokenize(q) if t not in STOPWORDS]
    return weighted_avg_embedding(q_tokens, idf).reshape(1, -1)

query_text = "spicy crab curry"
q_vec = embed_query_as_ingredients(query_text)

res = client.search(
    collection_name=collection_name,
    data=q_vec,
    limit=10,
    output_fields=["text", "recipe_id", "vector_type"],
    filter="vector_type == 'ingredients'"  # important: limit to ingredient vectors
)

# print top results
for hit in res[0]:
    print("recipe_id:", hit["entity"]["recipe_id"], "vector_type:", hit["entity"]["vector_type"], "distance:", hit["distance"])
    print(hit["entity"]["text"][:200], "...")
    print("----")
