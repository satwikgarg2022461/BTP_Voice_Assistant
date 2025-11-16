from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import requests
from dotenv import load_dotenv


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
        
        # Load API base URL for fetching full recipe details
        load_dotenv()
        self.api_base_url = os.getenv("API_BASE_URL")
        
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
    
    def fetch_full_recipe_details(self, recipe_id):
        """
        Fetch complete recipe details from the API
        
        Args:
            recipe_id (int): The recipe ID to fetch
            
        Returns:
            dict: Full recipe details including ingredients and instructions, or None on error
        """
        try:
            if not self.api_base_url:
                print("Warning: API_BASE_URL not configured")
                return None
                
            url = f"{self.api_base_url}/search-recipe/{recipe_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            payload = data
            recipe = {
                "recipe_id": payload["recipe"].get("recipe_id"),
                "title": payload["recipe"].get("recipe_title"),
                "ingredients": [
                    {
                        "ingredient": ing.get("ingredient"),
                        "state": ing.get("state"),
                        "quantity": ing.get("quantity"),
                        "unit": ing.get("unit"),
                        "ndb_id": ing.get("ndb_id"),
                    }
                    for ing in payload.get("ingredients", [])
                ],
                "process_tags": payload["recipe"].get("processes", "").split("||"),
                "metadata": {
                    "region": payload["recipe"].get("region"),
                    "sub_region": payload["recipe"].get("sub_region"),
                    "source": payload["recipe"].get("source"),
                    "url": payload["recipe"].get("url"),
                    "img_url": payload["recipe"].get("img_url"),
                    "nutritions": payload["recipe"].get("nutritions"),
                    "diet_flags": payload["recipe"].get("diet_flags"),
                }
            }
            
            # Fetch instructions separately from the instructions endpoint
            instructions = self.fetch_instructions(recipe_id)
            recipe["instructions"] = instructions
            
            return recipe
            
        except Exception as e:
            print(f"Error fetching full recipe details for ID {recipe_id}: {str(e)}")
            return None
    
    def fetch_instructions(self, recipe_id):
        """
        Fetch cooking instructions for a given recipe ID.
        
        Args:
            recipe_id (int): The recipe ID to fetch instructions for
            
        Returns:
            list: List of instruction steps
        """
        try:
            if not self.api_base_url:
                print("Warning: API_BASE_URL not configured")
                return []
                
            url = f"{self.api_base_url}/instructions/{recipe_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("steps", [])
            
        except Exception as e:
            print(f"Error fetching instructions for recipe ID {recipe_id}: {str(e)}")
            return []
    
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
                    full_text = hit["entity"]["text"]
                    formatted_results.append({
                        "recipe_id": hit["entity"]["recipe_id"],
                        "similarity": hit["distance"],  
                        "text_preview": full_text[:200] + "..." if len(full_text) > 200 else full_text,
                        "full_text": full_text,
                        "vector_type": hit["entity"]["vector_type"]
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching recipes: {str(e)}")
            return []