import pandas as pd
import re
import spacy


class RecipeNormalizer:
    """Class to handle recipe normalization and preparation for embeddings."""

    def __init__(self):
        """Initialize the normalizer with spaCy model."""
        self.nlp = spacy.load("en_core_web_sm")

    def lemmatize_text(self, text):
        """Lemmatize and remove stopwords/punctuation."""
        doc = self.nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

    def clean_ingredient_list(self, raw_ing):
        """Split ingredients string and drop quantities, keep names only."""
        ingredients = []
        for ing in raw_ing.split(","):
            ing = ing.strip().lower()
            # Remove numbers and fractions
            ing = re.sub(r"\d+(\.\d+)?(/\d+)?", "", ing)
            # Remove units (rough list)
            ing = re.sub(r"\b(tsp|tbsp|cup|cups|teaspoon|teaspoons|tablespoon|tablespoons|piece|pieces|pod|pods|wedge|grams?|ml|liter|kg)\b", "", ing)
            ing = re.sub(r"\s+", " ", ing).strip()
            if ing:
                ingredients.append(ing)
        return ingredients

    def parse_and_normalize(self, text, recipe_id=None):
        """
        Parse text of format:
        'Title | source: ... | region: ... | ingredients: ... | processes: ... | instructions: ... | nutrition: ...'
        """
        parts = {}
        for section in ["source", "region", "ingredients", "processes", "instructions", "nutrition"]:
            match = re.search(rf"{section}:(.*?)(?=\s\|\s\w+:|$)", text, re.IGNORECASE | re.DOTALL)
            if match:
                parts[section] = match.group(1).strip()

        title = text.split("|")[0].strip()

        # Normalize ingredients
        ingredients = []
        if "ingredients" in parts:
            ingredients = self.clean_ingredient_list(parts["ingredients"])

        # Normalize processes
        processes = []
        if "processes" in parts:
            processes = [self.lemmatize_text(p.strip()) for p in parts["processes"].split(",") if p.strip()]

        # Normalize instructions
        instructions = self.lemmatize_text(parts.get("instructions", ""))

        # Nutrition
        nutrition = {}
        if "nutrition" in parts:
            for item in parts["nutrition"].split(";"):
                if ":" in item:
                    k, v = item.split(":", 1)
                    val = v.strip()
                    nutrition[k.strip().lower()] = None if val.upper() == "NA" else val

        # Structured recipe
        normalized = {
            "recipe_id": recipe_id,
            "title": title.lower(),
            "source": parts.get("source", "").lower(),
            "region": parts.get("region", "").lower(),
            "ingredients": ingredients,
            "processes": processes,
            "instructions": instructions,
            "nutrition": nutrition
        }

        # Searchable text for embeddings
        searchable = f"{normalized['title']}. ingredients: {', '.join(ingredients)}."

        return normalized, searchable

    def process_recipes(self, input_csv="data/chunks.csv", output_csv="data/searchable_text_for_embeddings.csv"):
        """Process recipes and create searchable text for embeddings."""
        print("\n" + "="*60)
        print("STEP 2: Normalizing Recipes")
        print("="*60)

        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} recipes from {input_csv}")

        structured = []
        embedding_rows = []

        for _, row in df.iterrows():
            recipe_id = row["recipe_id"]
            text = str(row["searchable_text"])

            norm, searchable = self.parse_and_normalize(text, recipe_id)

            structured.append(norm)
            embedding_rows.append({"recipe_id": recipe_id, "searchable_text": searchable})

        # Save embeddings CSV
        pd.DataFrame(embedding_rows).to_csv(output_csv, index=False)

        print(f"✅ Normalization complete!")
        print(f"→ Embedding-ready CSV saved to {output_csv}")
        print("="*60 + "\n")

        return output_csv


# ---------- Run ----------
if __name__ == "__main__":
    normalizer = RecipeNormalizer()
    normalizer.process_recipes()
