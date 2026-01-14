import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient


class RecipeEmbeddings:
    """Class to handle recipe embeddings and vector database operations."""

    def __init__(self, db_path="recipes_demo.db", collection_name="recipes_collection"):
        """Initialize the embeddings generator with model and database."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim = self.model.get_sentence_embedding_dimension()
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None

    def tokenize(self, text):
        """Simple tokenization: words only."""
        return re.findall(r"\w+", text.lower())

    def extract_ingredients(self, text):
        """Extract ingredients section from text."""
        m = re.search(r"ingredients:(.*?)(instructions:|$)", text.lower(), re.DOTALL)
        if m:
            return m.group(1)
        # fallback: no ingredients label -> try short heuristic (first portion of text)
        return text[:200]

    def get_embedding(self, text):
        """Get standard embedding from sentence transformer."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def initialize_database(self):
        """Initialize Milvus client and create collection."""
        self.client = MilvusClient(self.db_path)

        # Drop existing collection if it exists
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)

        # Create new collection
        self.client.create_collection(collection_name=self.collection_name, dimension=self.dim)
        print(f"Created collection '{self.collection_name}' with dimension {self.dim}")

    def create_embeddings(self, input_csv="data/searchable_text_for_embeddings.csv"):
        """Create embeddings and insert into vector database."""
        print("\n" + "="*60)
        print("STEP 4: Creating Embeddings & Building Vector Database")
        print("="*60)

        # Load data
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} recipes from {input_csv}")

        # Initialize database
        self.initialize_database()

        # Build and insert embeddings
        entities = []
        global_id = 0

        for _, row in df.iterrows():
            rid = int(row["recipe_id"])
            text = str(row["searchable_text"])

            # Get full text embedding
            full_emb = self.get_embedding(text)

            # Insert one row per recipe with full embedding
            entities.append({
                "id": global_id,
                "vector": full_emb.tolist(),
                "text": text,
                "recipe_id": rid,
                "vector_type": "full"
            })
            global_id += 1

        # Bulk insert
        res = self.client.insert(collection_name=self.collection_name, data=entities)
        print(f"✅ Inserted {len(entities)} embeddings into vector database")
        print(f"→ Database saved to {self.db_path}")
        print("="*60 + "\n")

        return res

    def search(self, query_text, limit=10):
        """Search for similar recipes using query text."""
        if self.client is None:
            self.client = MilvusClient(self.db_path)

        q_vec = self.get_embedding(query_text).reshape(1, -1)

        res = self.client.search(
            collection_name=self.collection_name,
            data=q_vec,
            limit=limit,
            output_fields=["text", "recipe_id", "vector_type"]
        )

        return res


if __name__ == "__main__":
    embeddings = RecipeEmbeddings()
    embeddings.create_embeddings()

    # Test search
    print("\n" + "="*60)
    print("Testing Search")
    print("="*60)
    query_text = "I have ingredient like tiger prawn ounces. What to make"
    results = embeddings.search(query_text, limit=10)

    print(f"\nQuery: {query_text}\n")
    for hit in results[0]:
        print(f"Recipe ID: {hit['entity']['recipe_id']}, Vector Type: {hit['entity']['vector_type']}, Distance: {hit['distance']}")
        print(f"{hit['entity']['text'][:200]}...")
        print("----")
