import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from pymilvus import MilvusClient

client = MilvusClient("recipes_demo.db")
collection_name = "recipes_collection"

model = Word2Vec.load("models/Food2Vec.bin")
embedding_dim = model.vector_size  # should be 100

def get_text_embedding(text):
    tokens = text.lower().split()
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(embedding_dim, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)

query_text = "turmeric crab"
query_vector = get_text_embedding(query_text).reshape(1, -1)

res = client.search(
    collection_name=collection_name,
    data=query_vector,
    limit=3,
    output_fields=["text", "recipe_id"]
)

print("Search Results:\n", res)