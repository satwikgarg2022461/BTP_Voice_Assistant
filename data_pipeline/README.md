# Data Pipeline Documentation

This document describes the recipe data processing pipeline used to prepare data for the BTP Voice Assistant.

---

## Overview

The data pipeline transforms raw recipe data from an external API into the structured formats required by the Voice Assistant:
- **chunks.csv** - Segmented recipe instructions for navigation
- **food_dictionary.csv** - Ingredient lookup by recipe
- **embeddings** - Vectorized recipes for semantic search (Milvus)

---

## Pipeline Architecture

```
External Recipe API
        ↓
    api_call.py
        ↓
    Raw recipe JSON
    {recipe_id, title, ingredients, instructions}
        ↓
    normalizer.py
        ↓
    Normalized text
    (lowercased, cleaned, standardized units)
        ↓
    chunker.py
        ↓
    chunks.csv + searchable_text_for_embeddings.csv
        ↓
    Embedding generation
    (SentenceTransformer)
        ↓
    Milvus vector DB (recipes_demo.db)
        ↓
    Voice Assistant ready for semantic search
```

---

## Components

### 1. api_call.py

**Purpose:** Fetch recipes from backend API

**Main function:**
```python
def fetch_recipes(api_url: str, batch_size: int = 100) -> List[Dict]:
    """
    Fetch recipes from API endpoint
    
    Args:
        api_url: Base URL of recipe backend
        batch_size: Number of recipes per request
    
    Returns:
        List of raw recipe dicts:
        {
            "recipe_id": int,
            "recipe_title": str,
            "cuisine": str,
            "ingredients": [
                {"ingredient": str, "quantity": float, "unit": str},
                ...
            ],
            "instructions": [
                {"step_number": int, "step": str},
                ...
            ]
        }
    """
```

**API Endpoint:**
- `GET /recipes?offset=0&limit=100`
- Returns paginated recipe list

**Error handling:**
- Retry on timeout (3 attempts)
- Log failures, skip to next recipe
- Report summary at end

---

### 2. normalizer.py

**Purpose:** Clean and standardize recipe text

**Main function:**
```python
def normalize_recipe(recipe: Dict) -> Dict:
    ```
    Normalize recipe text:
    - Lowercase all text
    - Standardize units (cup → cup, teaspoon → tsp, etc.)
    - Remove special characters
    - Expand abbreviations (tbsp → tablespoon)
    - Remove redundant whitespace
    
    Returns:
        Normalized recipe dict
    ```
```

**Transformations:**

| Original | Normalized |
|----------|------------|
| "1 Tbsp olive oil" | "1 tablespoon olive oil" |
| "2    cloves  GARLIC" | "2 cloves garlic" |
| "Preheat oven @ 350°F" | "preheat oven at 350 degrees fahrenheit" |

**Reasons:**
- Case normalization: Ensures consistent embeddings
- Unit standardization: Prevents "tbsp" and "tablespoon" from being treated as different
- Whitespace cleanup: Improves embedding quality

---

### 3. chunker.py

**Purpose:** Divide recipes into navigable chunks

**Main function:**
```python
def chunk_recipe(recipe: Dict, chunk_size: int = 3) -> List[Dict]:
    """
    Split recipe instructions into chunks
    
    Args:
        recipe: Normalized recipe dict
        chunk_size: Target steps per chunk (3-5 recommended)
    
    Returns:
        List of chunk dicts:
        {
            "recipe_id": int,
            "title": str,
            "chunk_index": int,       # 1-based
            "start_step": int,        # global step number
            "end_step": int,
            "chunk_text": str,        # full text of steps
            "searchable_text": str,   # keywords for embedding
            "metadata": {}
        }
    """
```

**Algorithm:**
1. Parse instructions into sentences
2. Group by chunk_size steps (e.g., 3 steps per chunk)
3. Refine boundaries (don't split mid-instruction)
4. Generate searchable_text (ingredient keywords)

**Example:**

```
Recipe: Spaghetti Aglio e Olio (8 steps)

Instructions:
1. Bring a large pot of salted water to a boil.
2. Meanwhile, heat olive oil in a large skillet over low heat.
3. Add sliced garlic and red pepper flakes. Cook until fragrant, about 2 minutes.
4. Add the spaghetti to the boiling water and cook until al dente, about 8-10 minutes.
5. Reserve 1 cup of pasta water, then drain.
6. Add the drained pasta to the skillet with the garlic oil.
7. Toss well, adding pasta water as needed to create a silky sauce.
8. Serve immediately with grated Parmesan cheese.

Chunking (3 steps per chunk):
├─ Chunk 1 (index=1):
│   start_step: 1
│   end_step: 3
│   chunk_text: "Bring a large pot... Cook until fragrant..."
│   searchable_text: "water boil pot olive oil garlic heat fragrant"
│
├─ Chunk 2 (index=2):
│   start_step: 4
│   end_step: 6
│   chunk_text: "Add the spaghetti... pasta water drain..."
│   searchable_text: "spaghetti cook al dente water drain combine oil"
│
└─ Chunk 3 (index=3):
    start_step: 7
    end_step: 8
    chunk_text: "Toss well... Serve immediately..."
    searchable_text: "toss sauce water silky serve parmesan"
```

---

### 4. main.py (Pipeline Orchestrator)

**Purpose:** Coordinate entire pipeline

**Main function:**
```python
def run_pipeline(
    api_url: str,
    output_dir: str = "data",
    chunk_size: int = 3,
    num_recipes: int = None  # None = all
):
    """
    1. Fetch raw recipes from API
    2. Normalize each recipe
    3. Chunk each recipe
    4. Write CSVs
    5. Generate embeddings for Milvus
    6. Insert into Milvus DB
    """
```

**Workflow:**
```python
recipes = api_call.fetch_recipes(api_url)

for recipe in recipes:
    # 1. Normalize
    recipe = normalizer.normalize_recipe(recipe)
    
    # 2. Chunk
    chunks = chunker.chunk_recipe(recipe, chunk_size)
    
    # 3. Extract ingredients
    ingredients = extract_ingredients(recipe)
    
    # 4. Write to CSVs
    write_chunks_csv(chunks)
    write_food_dictionary_csv(recipe_id, ingredients)

# 5. Generate embeddings
embeddings = generate_embeddings(
    df_searchable_text,
    model='all-MiniLM-L6-v2'
)

# 6. Insert into Milvus
insert_to_milvus(embeddings, recipe_ids)
```

---

## Output Formats

### chunks.csv

```csv
recipe_id,title,chunk_index,start_step,end_step,chunk_text,searchable_text,metadata
4521,Spaghetti Aglio e Olio,1,1,3,"Bring a large pot...","water boil pot olive oil","{}"
4521,Spaghetti Aglio e Olio,2,4,6,"Add the spaghetti...","spaghetti cook al dente","{}"
4521,Spaghetti Aglio e Olio,3,7,8,"Toss well...","toss sauce silky","{}"
```

### food_dictionary.csv

```csv
recipe_id,recipe_name,ingredient,quantity,unit,ndb_id
4521,Spaghetti Aglio e Olio,spaghetti,1,pound,20420
4521,Spaghetti Aglio e Olio,garlic,6,cloves,11215
4521,Spaghetti Aglio e Olio,olive oil,0.5,cup,4053
```

### searchable_text_for_embeddings.csv

```csv
recipe_id,searchable_text,embedding_vector
4521,"spaghetti aglio olio pasta garlic olive oil recipe",[0.12,-0.45,...]
```

---

## Processing Statistics

**Typical run for 5000 recipes:**

| Stage | Time | Input | Output |
|-------|------|-------|--------|
| Fetch API | ~5 min | 5000 recipe IDs | 5000 JSON objects (~10 MB) |
| Normalize | ~30 sec | 5000 recipes | Cleaned text |
| Chunk | ~2 min | 5000 recipes | ~15K chunks (avg 3 per recipe) |
| Write CSVs | ~1 min | Chunks + ingredients | 50 MB CSVs |
| Embed | ~3 min | 15K searchable texts | 15K × 384-D vectors (22 MB) |
| Insert to Milvus | ~2 min | Embeddings + IDs | Milvus index built |
| **Total** | **~13 min** | **API config** | **Ready for search** |

---

## Error Handling

### API Failures

```python
try:
    recipes = api_call.fetch_recipes(...)
except RequestException as e:
    log.error(f"API error: {e}")
    # Retry with backoff
    retry_count = 0
    while retry_count < 3:
        time.sleep(2 ** retry_count)
        try:
            recipes = api_call.fetch_recipes(...)
            break
        except RequestException:
            retry_count += 1
    if retry_count == 3:
        raise Exception("API permanently unavailable")
```

### Invalid Data

```python
for recipe in recipes:
    if not recipe.get('ingredients') or not recipe.get('instructions'):
        log.warning(f"Skipping recipe {recipe['recipe_id']}: missing fields")
        continue
    
    # Validate chunk count
    if len(chunks) > 20:  # Too many chunks (recipe too long)
        log.warning(f"Recipe {recipe['recipe_id']}: too many chunks, truncating")
        chunks = chunks[:20]
```

---

## Deployment

### Prerequisites

```bash
pip install requests pandas sentence-transformers pymilvus
```

### Run Pipeline

```bash
# Method 1: Python script
python data_pipeline/main.py \
    --api_url "http://localhost:5000" \
    --output_dir "./data" \
    --chunk_size 3

# Method 2: As module
from data_pipeline import main
main.run_pipeline(
    api_url="http://localhost:5000",
    num_recipes=5000
)
```

### Verify Output

```bash
# Check CSV generation
wc -l data/chunks.csv            # Should have ~5000*3 lines
head -5 data/food_dictionary.csv

# Check Milvus index
python -c "
from pymilvus import MilvusClient
client = MilvusClient('recipes_demo.db')
print(f'Collection size: {client.num_entities(「recipes_collection")}')"
```

---

## Updating Recipes

**Scenario:** Add new recipes to the system

**Steps:**
1. New recipes available in API
2. Run pipeline (fetches only new recipes via offset)
3. Newly generated chunks + embeddings inserted to Milvus
4. Voice Assistant automatically has access to new recipes (on next restart)

**Note:** Pipeline is **idempotent** (safe to re-run; won't duplicate)

---

## Data Quality Checks

### Pre-Pipeline Validation

```python
def validate_recipe(recipe: Dict) -> bool:
    checks = [
        len(recipe.get('recipe_title', '')) > 3,
        len(recipe.get('ingredients', [])) > 0,
        len(recipe.get('instructions', [])) > 0,
        all(0 < len(i['ingredient']) < 100 for i in recipe.get('ingredients', [])),
        all(0 < len(s['step']) < 500 for s in recipe.get('instructions', [])),
    ]
    return all(checks)
```

### Post-CSVization Validation

```python
def validate_output():
    # Check no duplicate recipe_ids
    assert chunks_df['recipe_id'].nunique() == len(chunks_df['recipe_id'].unique())
    
    # Check step continuity
    for recipe_id in chunks_df['recipe_id'].unique():
        recipe_chunks = chunks_df[chunks_df['recipe_id'] == recipe_id]
        steps = recipe_chunks['start_step'].tolist()
        assert steps == sorted(steps), f"Out of order steps for recipe {recipe_id}"
    
    # Check embedding dimension
    assert embeddings.shape[1] == 384, "Embedding dim != 384"
```

---

## Performance Optimization

### Parallelization (Future)

```python
# Currently: sequential processing
for recipe in recipes:
    recipe = normalize(recipe)
    chunks = chunk(recipe)
    write_csv(chunks)

# Future: parallel processing
from multiprocessing import Pool

with Pool(processes=4) as pool:
    normalized = pool.map(normalize, recipes)
    chunks = pool.map(chunk, normalized)
```

### Caching

```python
# Cache normalized recipes
import pickle

if os.path.exists('cache/normalized.pkl'):
    normalized = pickle.load(open('cache/normalized.pkl', 'rb'))
else:
    normalized = [normalize(r) for r in recipes]
    pickle.dump(normalized, open('cache/normalized.pkl', 'wb'))
```

---

## Example: Complete Data Flow

```
User query: "show me how to make pasta"
    ↓
[Voice Assistant starts]
    ↓
[Retriever embeds query]
    ↓
Embedding: [0.12, -0.45, 0.03, ...]  (384-D)
    ↓
[Milvus search]
    ↓
Top-5 similar recipes:
  1. Spaghetti Aglio e Olio (similarity: 0.892)
  2. Fettuccine Alfredo (0.856)
  3. Pasta Carbonara (0.843)
  ...
    ↓
[Select top result] → recipe_id = 4521
    ↓
[Fetch from searchable_text_for_embeddings.csv]
    ↓
[Load full recipe via API] → ingredients + full instructions
    ↓
[Generate LLM response]
    ↓
User hears: "Let me show you Spaghetti Aglio e Olio..."
```

---

## Maintenance

### Regenerating Data

When to regenerate:
- New recipes added to backend (daily/weekly)
- Normalization rules updated (after code change)
- Chunk size tuning (change chunk_size parameter)

### Archiving Old Data

```bash
# Keep last 3 versions
cd data
ls -t *.csv | tail -n +4 | xargs rm  # Delete older than 3
```

---

This data pipeline enables the Voice Assistant to intelligently navigate and present recipe information at scale.
