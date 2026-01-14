import os
import time
from api_call import RecipeAPI
from chunker import RecipeChunker
from normalizer import RecipeNormalizer
from food_dictionary import FoodDictionary
from embeddings import RecipeEmbeddings


def main():
    """
    Complete data pipeline:
    STEP 1: Fetch recipes from API and create chunks
    STEP 2: Normalize recipes
    STEP 3: Create food dictionary
    STEP 4: Create embeddings and build vector database
    """

    # Initialize all pipeline components
    api = RecipeAPI()
    chunker = RecipeChunker()
    normalizer = RecipeNormalizer()
    food_dict = FoodDictionary()
    embeddings = RecipeEmbeddings()

    # Files for the pipeline
    recipes_csv = "recipes.csv"
    chunks_csv = "chunks.csv"
    searchable_csv = "searchable_text_for_embeddings.csv"
    food_dict_csv = "food_dictionary.csv"

    print("\n" + "="*60)
    print("STARTING RECIPE DATA PIPELINE")
    print("="*60 + "\n")

    # ==================== STEP 1: Fetch Recipes ====================
    print("="*60)
    print("STEP 1: Fetching Recipes from API")
    print("="*60)

    # Check if files already exist to determine if we're resuming
    resuming = os.path.exists(os.path.join(chunker.output_dir, recipes_csv))
    
    print("Fetching recipes...")
    
    # Fetch recipes across multiple pages until we have at least 100
    all_recipes_meta = []
    page = 1
    target_count = 100
    
    while len(all_recipes_meta) < target_count:
        print(f"Fetching page {page}...")
        try:
            recipes_meta = api.fetch_recipes("Indian Subcontinent", "Indian", page=page)
            if not recipes_meta:  # No more results
                print(f"No more recipes available after page {page-1}")
                break
                
            all_recipes_meta.extend(recipes_meta)
            print(f"Found {len(recipes_meta)} recipes on page {page}. Total: {len(all_recipes_meta)}")
            page += 1
            
            # Small delay to be kind to the API server
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error fetching page {page}: {str(e)}")
            break
    
    # Slice to get exactly 100 recipes (or all if less than 100)
    total_recipes = min(target_count, len(all_recipes_meta))
    all_recipes_meta = all_recipes_meta[:total_recipes]
    
    print(f"Found a total of {len(all_recipes_meta)} recipes, will process {total_recipes}")
    
    # Create empty files with headers if not resuming
    if not resuming:
        chunker.export_recipes_csv([], recipes_csv)
        chunker.export_chunks_csv([], chunks_csv)
        print("Created CSV files with headers")
        processed_count = 0
    else:
        # Count existing recipes to determine progress
        try:
            with open(os.path.join(chunker.output_dir, recipes_csv), 'r') as f:
                # Subtract 1 for the header row
                processed_count = sum(1 for _ in f) - 1
            print(f"Resuming from recipe {processed_count + 1}/{total_recipes}")
        except Exception as e:
            print(f"Error reading existing files: {str(e)}")
            processed_count = 0
    
    # Process recipes incrementally
    start_time = time.time()
    batch_size = 5  # Save after every 5 recipes
    current_batch_recipes = []
    current_batch_chunks = []
    
    for i, recipe_meta in enumerate(all_recipes_meta[:total_recipes], 1):
        # Skip already processed recipes if resuming
        if i <= processed_count:
            continue
            
        rid = recipe_meta["recipe_id"]
        
        # Calculate and display progress
        elapsed_time = time.time() - start_time
        recipes_per_sec = i / elapsed_time if elapsed_time > 0 else 0
        remaining = total_recipes - i
        estimated_time = remaining / recipes_per_sec if recipes_per_sec > 0 else 0
        
        print(f"\n[{i}/{total_recipes}] ({(i/total_recipes)*100:.1f}%) " +
              f"Fetching Recipe ID {rid}... " +
              f"ETA: {int(estimated_time//60):02d}:{int(estimated_time%60):02d}")
        
        try:
            # Get recipe and chunks
            full_recipe = api.get_full_recipe(rid)
            chunks = chunker.chunk_instructions(full_recipe)
            
            # Add to current batch
            current_batch_recipes.append(full_recipe)
            current_batch_chunks.extend(chunks)
            
            print(f"Recipe '{full_recipe['title']}' processed with {len(chunks)} chunks")
            
            # Save incrementally after each batch
            if i % batch_size == 0 or i == total_recipes:
                # Append to existing files
                chunker.export_recipes_csv(current_batch_recipes, recipes_csv, append=True)
                chunker.export_chunks_csv(current_batch_chunks, chunks_csv, append=True)
                
                print(f"✓ Saved batch ({len(current_batch_recipes)} recipes, {len(current_batch_chunks)} chunks)")
                
                # Clear batch after saving
                current_batch_recipes = []
                current_batch_chunks = []
        except Exception as e:
            print(f"Error processing recipe {rid}: {str(e)}")
            # Continue with next recipe on error
            
    total_time = time.time() - start_time
    print(f"\n✅ STEP 1 Completed in {int(total_time//60):02d}:{int(total_time%60):02d}")
    print(f"✅ Processed {total_recipes} recipes")
    print(f"✅ Data saved to {chunker.output_dir}/{recipes_csv} and {chunker.output_dir}/{chunks_csv}")
    print("="*60 + "\n")

    # ==================== STEP 2: Normalize Recipes ====================
    try:
        normalizer.process_recipes(
            input_csv=os.path.join(chunker.output_dir, chunks_csv),
            output_csv=os.path.join(chunker.output_dir, searchable_csv)
        )
    except Exception as e:
        print(f"❌ Error in normalization: {str(e)}")
        raise

    # ==================== STEP 3: Create Food Dictionary ====================
    try:
        food_dict.create_food_dictionary(
            input_csv=os.path.join(chunker.output_dir, searchable_csv),
            output_csv=os.path.join(chunker.output_dir, food_dict_csv)
        )
    except Exception as e:
        print(f"❌ Error creating food dictionary: {str(e)}")
        raise

    # ==================== STEP 4: Create Embeddings ====================
    try:
        embeddings.create_embeddings(
            input_csv=os.path.join(chunker.output_dir, searchable_csv)
        )
    except Exception as e:
        print(f"❌ Error creating embeddings: {str(e)}")
        raise

    # ==================== Pipeline Complete ====================
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETE!")
    print("="*60)
    print(f"✅ Recipes CSV: {chunker.output_dir}/{recipes_csv}")
    print(f"✅ Chunks CSV: {chunker.output_dir}/{chunks_csv}")
    print(f"✅ Searchable Text CSV: {chunker.output_dir}/{searchable_csv}")
    print(f"✅ Food Dictionary CSV: {chunker.output_dir}/{food_dict_csv}")
    print(f"✅ Vector Database: recipes_demo.db")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
