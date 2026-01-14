import pandas as pd
import re


class FoodDictionary:
    """Class to create a food dictionary from recipe data."""

    def __init__(self):
        """Initialize the food dictionary creator."""
        pass

    def extract_recipe_name(self, text):
        """Extract recipe name from the beginning of the searchable text."""
        # Extract text up to the first period or dash
        match = re.match(r'^(.*?)[.-]', text)
        if match:
            return match.group(1).strip()
        # If no period or dash, take the first 30 chars or up to "ingredients:" if present
        ingredients_idx = text.lower().find('ingredients:')
        if ingredients_idx > 0:
            return text[:ingredients_idx].strip()
        return text[:30].strip() + '...'

    def extract_ingredients(self, text):
        """Extract ingredients from text."""
        # Look for "ingredients:" section in the text
        match = re.search(r'ingredients:(.*?)(?:instructions:|$)', text.lower(), re.DOTALL)
        if match:
            # Get the ingredients text and clean it up
            ingredients_text = match.group(1).strip()
            # Split by commas and clean each ingredient
            ingredients = [ing.strip() for ing in ingredients_text.split(',')]
            return ingredients
        return []

    def create_food_dictionary(self, input_csv="data/searchable_text_for_embeddings.csv",
                               output_csv="data/food_dictionary.csv"):
        """Create a food dictionary CSV from searchable text data."""
        print("\n" + "="*60)
        print("STEP 3: Creating Food Dictionary")
        print("="*60)

        try:
            df = pd.read_csv(input_csv)
            print(f"Loaded {len(df)} recipes from {input_csv}")

            # Create a list to hold the processed data
            recipes_data = []

            for _, row in df.iterrows():
                recipe_id = row["recipe_id"]
                text = row["searchable_text"]

                # Extract recipe name and ingredients
                recipe_name = self.extract_recipe_name(text)
                ingredients = self.extract_ingredients(text)
                ingredients_str = ", ".join(ingredients)

                # Add to data list
                recipes_data.append({
                    "recipe_id": recipe_id,
                    "recipe_name": recipe_name,
                    "ingredients": ingredients_str
                })

            # Create a new DataFrame and save to CSV
            output_df = pd.DataFrame(recipes_data)
            output_df.to_csv(output_csv, index=False)
            print(f"✅ Successfully created {output_csv} with {len(output_df)} recipes")

            # Preview a few rows
            print("\nPreview of food dictionary:")
            print(output_df.head(3))
            print("="*60 + "\n")

            return output_csv

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            raise


if __name__ == "__main__":
    food_dict = FoodDictionary()
    food_dict.create_food_dictionary()
