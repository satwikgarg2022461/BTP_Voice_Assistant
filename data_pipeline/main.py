from api_call import RecipeAPI
from chunker import RecipeChunker


def main():
    api = RecipeAPI()
    chunker = RecipeChunker()

    print("Fetching recipes list...")
    recipes_meta = api.fetch_recipes("Indian Subcontinent", "Indian", page=1)

    all_recipes = []
    all_chunks = []

    for recipe_meta in recipes_meta[:2]:  # just demo on 2 recipes
        rid = recipe_meta["recipe_id"]
        print(f"\nFetching full details for Recipe ID {rid}...")
        full_recipe = api.get_full_recipe(rid)
        all_recipes.append(full_recipe)

        chunks = chunker.chunk_instructions(full_recipe)
        all_chunks.extend(chunks)

    # Save CSVs
    chunker.export_recipes_csv(all_recipes, "recipes.csv")
    chunker.export_chunks_csv(all_chunks, "chunks.csv")

    print("\n✅ Exported recipes.csv and chunks.csv")


if __name__ == "__main__":
    main()
