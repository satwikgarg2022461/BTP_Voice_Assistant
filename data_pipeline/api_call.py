import os
import requests
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class RecipeAPI:
    def __init__(self):
        self.base_url = os.getenv("API_BASE_URL")
        if not self.base_url:
            raise ValueError("API_BASE_URL not found in .env file")

    def fetch_recipes(self, cuisine: str, sub_region: str, page: int = 1):
        """
        Fetch a list of recipes for a given cuisine and sub-region.
        """
        url = f"{self.base_url}/recipes_cuisine/cuisine/{cuisine}"
        params = {"subRegion": sub_region, "page": page}

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise ValueError("Failed to fetch recipes")

        return data["payload"]["data"]

    def fetch_recipe_details(self, recipe_id: int):
        """
        Fetch detailed recipe information by recipe ID.
        """
        url = f"{self.base_url}/search-recipe/{recipe_id}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        payload = data
        # print(data)
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
        return recipe

    def fetch_instructions(self, recipe_id: int):
        """
        Fetch cooking instructions for a given recipe ID.
        """
        url = f"{self.base_url}/instructions/{recipe_id}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("steps", [])

    def get_full_recipe(self, recipe_id: int):
        """
        Fetch full recipe details including ingredients and instructions.
        """
        details = self.fetch_recipe_details(recipe_id)
        instructions = self.fetch_instructions(recipe_id)
        details["instructions"] = instructions
        return details
