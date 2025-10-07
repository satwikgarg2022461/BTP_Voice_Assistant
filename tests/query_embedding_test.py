import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

client = MilvusClient("recipes_demo.db")
collection_name = "recipes_collection"

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = model.get_sentence_embedding_dimension()

def get_embedding(text):
    # Get standard embedding from sentence transformer
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.astype(np.float32)

# ---------- searching ----------
def embed_query(query_text):
    return get_embedding(query_text).reshape(1, -1)

query_text = "How to make spicy crap curry"
q_vec = embed_query(query_text)

res = client.search(
    collection_name=collection_name,
    data=q_vec,
    limit=10,
    output_fields=["text", "recipe_id", "vector_type"]
)
print('\n')
print("==================Results==============================\n")
# print top results
for hit in res[0]:
    print("recipe_id:", hit["entity"]["recipe_id"], "vector_type:", hit["entity"]["vector_type"], "distance:", hit["distance"])
    print(hit["entity"]["text"][:200], "...")
    print("----")
