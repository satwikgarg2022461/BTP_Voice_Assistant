from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import numpy as np


class RecipeRetriever:
    def __init__(self, db_path="recipes_demo.db", collection_name="recipes_collection", model_name='all-MiniLM-L6-v2'):
        """
        Initialize the recipe retriever with Milvus client and embedding model
        
        Args:
            db_path (str): Path to the Milvus database
            collection_name (str): Name of the Milvus collection
            model_name (str): Name of the sentence transformer model
        """
        # Initialize Milvus client
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        
        # Load the embedding model
        self.model = SentenceTransformer(model_name)
        
        # Check if collection exists
        if not self.client.has_collection(collection_name=self.collection_name):
            print(f"Warning: Collection '{self.collection_name}' does not exist in the database.")
    
    def embed_query(self, query_text):
        """
        Generate embeddings for a query text
        
        Args:
            query_text (str): The text to embed
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        embedding = self.model.encode(query_text, normalize_embeddings=True)
        return embedding.astype(np.float32).reshape(1, -1)
    
    def search_recipes(self, query_text, limit=5):
        """
        Search for recipes based on the query text
        
        Args:
            query_text (str): The query text
            limit (int): Maximum number of results to return
            
        Returns:
            list: List of recipe results with their metadata and similarity scores
        """
        try:
            # Generate query embedding
            query_vector = self.embed_query(query_text)
            
            # Search in Milvus
            results = self.client.search(
                collection_name=self.collection_name,
                data=query_vector,
                limit=limit,
                output_fields=["text", "recipe_id", "vector_type"]
            )
            
            # Format results
            formatted_results = []
            if results and len(results) > 0:
                for hit in results[0]:
                    formatted_results.append({
                        "recipe_id": hit["entity"]["recipe_id"],
                        "similarity": hit["distance"],  
                        "text_preview": hit["entity"]["text"][:200] + "..." if len(hit["entity"]["text"]) > 200 else hit["entity"]["text"],
                        "vector_type": hit["entity"]["vector_type"]
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching recipes: {str(e)}")
            return []