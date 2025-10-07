import pandas as pd
import re
import json
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ---------- Helpers ----------

def lemmatize_text(text):
    """Lemmatize and remove stopwords/punctuation."""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def clean_ingredient_list(raw_ing):
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

def parse_and_normalize(text, recipe_id=None):
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
        ingredients = clean_ingredient_list(parts["ingredients"])

    # Normalize processes
    processes = []
    if "processes" in parts:
        processes = [lemmatize_text(p.strip()) for p in parts["processes"].split(",") if p.strip()]

    # Normalize instructions
    instructions = lemmatize_text(parts.get("instructions", ""))

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


# ---------- Main Pipeline ----------

def process_recipes(input_csv="data/chunks.csv"):
    df = pd.read_csv(input_csv)

    structured = []
    embedding_rows = []

    for _, row in df.iterrows():
        recipe_id = row["recipe_id"]
        text = str(row["searchable_text"])

        norm, searchable = parse_and_normalize(text, recipe_id)

        structured.append(norm)
        embedding_rows.append({"recipe_id": recipe_id, "searchable_text": searchable})

    # Save structured JSONL
    # with open("data/chunks_normalized.jsonl", "w", encoding="utf-8") as f:
    #     for rec in structured:
    #         f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Save embeddings CSV
    pd.DataFrame(embedding_rows).to_csv("data/searchable_text_for_embeddings.csv", index=False)

    print("✅ Processing complete!")
    # print(f"→ Structured recipes saved to data/chunks_normalized.jsonl")
    print(f"→ Embedding-ready CSV saved to data/searchable_text_for_embeddings.csv")


# ---------- Run ----------
if __name__ == "__main__":
    process_recipes()
