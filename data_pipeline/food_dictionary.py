import pandas as pd
import re
import csv

def extract_recipe_name(text):
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

def extract_ingredients(text):
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

def create_food_dictionary():
    """Create a food dictionary CSV from searchable text data."""
    # Read the searchable text CSV
    input_path = "data/searchable_text_for_embeddings.csv"
    output_path = "data/food_dictionary.csv"
    
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} recipes from {input_path}")
        
        # Create a list to hold the processed data
        recipes_data = []
        
        for _, row in df.iterrows():
            recipe_id = row["recipe_id"]
            text = row["searchable_text"]
            
            # Extract recipe name and ingredients
            recipe_name = extract_recipe_name(text)
            ingredients = extract_ingredients(text)
            ingredients_str = ", ".join(ingredients)
            
            # Add to data list
            recipes_data.append({
                "recipe_id": recipe_id,
                "recipe_name": recipe_name,
                "ingredients": ingredients_str
            })
        
        # Create a new DataFrame and save to CSV
        output_df = pd.DataFrame(recipes_data)
        output_df.to_csv(output_path, index=False)
        print(f"Successfully created {output_path} with {len(output_df)} recipes")
        
        # Preview a few rows
        print("\nPreview of food dictionary:")
        print(output_df.head(3))
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    create_food_dictionary()