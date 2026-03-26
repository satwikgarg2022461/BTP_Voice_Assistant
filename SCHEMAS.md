# Data Schemas & Specifications

This document formally defines all major data structures, JSON schemas, and CSV formats used throughout the BTP Voice Assistant.

---

## 1. ASR Output Schema

### Deepgram ASR Response

**Endpoint:** `POST https://api.deepgram.com/v1/listen`

**Output Format:** JSON

```json
{
  "metadata": {
    "transaction_key": "string",
    "request_id": "string",
    "sha256": "string",
    "created": "2026-03-19T10:30:00Z",
    "duration": 4.32,
    "channels": 1,
    "models": ["base"],
    "model_info": {}
  },
  "results": {
    "channels": [
      {
        "alternatives": [
          {
            "confidence": 0.95,
            "transcript": "how do i make pasta"
          }
        ]
      }
    ],
    "final": true
  }
}
```

### Stored Transcription (ASR_text/)

**File:** `ASR_text/recording_20260319_103045.txt`

```
how do i make pasta
```

(Plain text, one line, lowercased)

---

## 2. Intent Classification Schema

### Classification Result

**Type:** Tuple returned by `IntentClassifier.classify()`

```python
(
    intent: Intent,           # Enum value
    confidence: float,        # 0.0-1.0
    entities: Dict[str, Any]
)
```

### Intent Enum

```python
class Intent(Enum):
    NAV_NEXT = "nav_next"
    NAV_PREV = "nav_prev"
    NAV_GO_TO = "nav_go_to"
    NAV_REPEAT = "nav_repeat"
    NAV_REPEAT_INGREDIENTS = "nav_repeat_ingredients"
    NAV_START = "nav_start"
    QUESTION = "question"
    SEARCH_RECIPE = "search_recipe"
    START_RECIPE = "start_recipe"
    STOP_PAUSE = "stop_pause"
    RESUME = "resume"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    SMALL_TALK = "small_talk"
    CLARIFY = "clarify"
    HELP = "help"
    UNKNOWN = "unknown"
```

### Extracted Entities

**Examples:**

```python
# For NAV_GO_TO intent
{
    "target_step": 5,
    "target_section": "ingredients"  # optional
}

# For QUESTION intent with ingredient substitution
{
    "ingredient": "salt",
    "question_type": "substitute"  # or "quantity", "time", "method", "why"
}

# For SEARCH_RECIPE intent
{
    "dish_name": "pasta"
}
```

---

## 3. Recipe Retrieval Schema

### Search Query Input

```
User input: "How do I make pasta?"
Vectorized: 384-D float32 embedding from SentenceTransformer
```

### Milvus Search Result

**Structure:** List returned by `RecipeRetriever.search_recipes()`

```python
[
    {
        "recipe_id": 4521,
        "title": "Spaghetti Aglio e Olio",
        "similarity_score": 0.892,
        "metadata": {}
    },
    {
        "recipe_id": 2134,
        "title": "Pasta Carbonara",
        "similarity_score": 0.856,
        "metadata": {}
    },
    ...
]
```

### Full Recipe Details

**Source:** API response from `{API_BASE_URL}/search-recipe/{recipe_id}`

```json
{
  "recipe": {
    "recipe_id": 4521,
    "recipe_title": "Spaghetti Aglio e Olio",
    "cuisine": "Italian",
    "cook_time": 10,
    "prep_time": 5,
    "servings": 4
  },
  "ingredients": [
    {
      "ingredient": "spaghetti",
      "quantity": 1,
      "unit": "pound",
      "state": "dry",
      "ndb_id": 20420
    },
    {
      "ingredient": "garlic",
      "quantity": 6,
      "unit": "cloves",
      "state": "fresh",
      "ndb_id": 11215
    },
    {
      "ingredient": "olive oil",
      "quantity": 0.5,
      "unit": "cup",
      "state": "liquid",
      "ndb_id": 4053
    },
    {
      "ingredient": "salt",
      "quantity": 2,
      "unit": "teaspoons",
      "state": "solid",
      "ndb_id": 2047
    },
    {
      "ingredient": "red pepper flakes",
      "quantity": 0.25,
      "unit": "teaspoon",
      "state": "dry",
      "ndb_id": 2010
    }
  ],
  "instructions": [
    {
      "step_number": 1,
      "step": "Bring a large pot of salted water to a boil."
    },
    {
      "step_number": 2,
      "step": "Meanwhile, heat olive oil in a large skillet over low heat."
    },
    {
      "step_number": 3,
      "step": "Add sliced garlic and red pepper flakes. Cook until fragrant, about 2 minutes."
    },
    {
      "step_number": 4,
      "step": "Add the spaghetti to the boiling water and cook until al dente, about 8-10 minutes."
    },
    {
      "step_number": 5,
      "step": "Reserve 1 cup of pasta water, then drain."
    },
    {
      "step_number": 6,
      "step": "Add the drained pasta to the skillet with the garlic oil."
    },
    {
      "step_number": 7,
      "step": "Toss well, adding pasta water as needed to create a silky sauce."
    },
    {
      "step_number": 8,
      "step": "Serve immediately with grated Parmesan cheese."
    }
  ]
}
```

---

## 4. LLM Response Schema

### Structured JSON Response

**Type:** Returned by `RecipeLLM.generate_recipe_response()` (when `return_json=True`)

```json
{
  "greeting": "Let me show you how to make Spaghetti Aglio e Olio, an authentic Italian classic!",
  "ingredients": [
    {
      "text": "One pound of spaghetti pasta",
      "spoken": false
    },
    {
      "text": "Six fresh garlic cloves",
      "spoken": false
    },
    {
      "text": "Half a cup of extra virgin olive oil",
      "spoken": false
    },
    {
      "text": "Two teaspoons of salt",
      "spoken": false
    },
    {
      "text": "A quarter teaspoon of red pepper flakes",
      "spoken": false
    }
  ],
  "steps": [
    {
      "step_num": 1,
      "text": "Start by bringing a large pot of salted water to a boil.",
      "spoken": false
    },
    {
      "step_num": 2,
      "text": "While the water is heating, pour half a cup of olive oil into a large skillet and place it over low heat.",
      "spoken": false
    },
    {
      "step_num": 3,
      "text": "Slice your six garlic cloves thinly and add them to the warm oil along with the red pepper flakes.",
      "spoken": false
    },
    {
      "step_num": 4,
      "text": "Let this cook for about two minutes until the garlic becomes fragrant and golden.",
      "spoken": false
    },
    {
      "step_num": 5,
      "text": "Now add the spaghetti to your boiling salted water and cook it according to package directions until al dente, usually about eight to ten minutes.",
      "spoken": false
    },
    {
      "step_num": 6,
      "text": "Before draining, reserve one cup of the starchy pasta water.",
      "spoken": false
    },
    {
      "step_num": 7,
      "text": "Drain your pasta and add it directly to the skillet with the garlic and oil.",
      "spoken": false
    },
    {
      "step_num": 8,
      "text": "Toss everything together well, gradually adding pasta water as needed until you achieve a glossy, silky sauce.",
      "spoken": false
    }
  ],
  "closing": "Your homemade Spaghetti Aglio e Olio is ready to serve! Enjoy this delicious Italian dish, and garnish with fresh Parmesan cheese if desired."
}
```

### Validation Rules

**Required Fields:**
- `greeting`: Non-empty string
- `ingredients`: List of dicts with `text` (string) and `spoken` (boolean)
- `steps`: List of dicts with `step_num` (int ≥ 1), `text` (string), `spoken` (boolean)
- `closing`: Non-empty string

**Data Types:**
- `step_num` must be sequential (1, 2, 3, ...)
- All text fields should be 1-4 sentences
- `spoken` booleans initially false

---

## 5. Session State Schema

### Redis Session Object

**Key:** `session:{session_id}` (e.g., `session:user_001`)

**Type:** JSON (serialized)

**TTL:** 3600 seconds (configurable)

```json
{
  "session_id": "user_001",
  "recipe_id": "4521",
  "recipe_title": "Spaghetti Aglio e Olio",
  "current_step": 1,
  "current_chunk": 1,
  "total_steps": 8,
  "total_chunks": 3,
  "created_at": "2026-03-19T10:30:00Z",
  "last_updated": "2026-03-19T10:35:22Z",
  "last_activity": "2026-03-19T10:35:22Z",
  "is_paused": false,
  "paused_at": null,
  "response_structure": {
    "greeting": "...",
    "ingredients": [...],
    "steps": [...],
    "closing": "..."
  },
  "ingredients_spoken": [true, true, false, false, false],
  "steps_spoken": [true, false, false, false, false, false, false, false],
  "conversation_history": "conversation_history:user_001",
  "user_preferences": {
    "tts_speed": 1.0,
    "dietary_restrictions": []
  },
  "metadata": {
    "cuisine": "Italian",
    "difficulty": "easy",
    "cook_time": 10,
    "prep_time": 5
  }
}
```

### Conversation History (Redis)

**Key:** `conversation_history:{session_id}`

**Type:** Ordered list (Redis LPUSH/LRANGE)

**Format:**

```python
[
    {
        "role": "user",
        "content": "how do i make pasta?",
        "timestamp": "2026-03-19T10:30:15Z"
    },
    {
        "role": "assistant",
        "content": "Let me show you how to make Spaghetti Aglio e Olio...",
        "timestamp": "2026-03-19T10:30:20Z"
    },
    {
        "role": "user",
        "content": "what's the next step?",
        "timestamp": "2026-03-19T10:30:45Z"
    },
    {
        "role": "assistant",
        "content": "While the water is heating, pour half a cup of olive oil into a large skillet...",
        "timestamp": "2026-03-19T10:30:50Z"
    }
]
```

**Max history length:** 10 turns (configurable)

---

## 6. CSV Data Formats

### chunks.csv

**Location:** `data/chunks.csv`

**Purpose:** Recipe segmentation for efficient navigation

**Schema:**

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| recipe_id | int | 4521 | Unique recipe identifier |
| title | string | Spaghetti Aglio e Olio | Recipe name |
| chunk_index | int | 1 | 1-based chunk number within recipe |
| start_step | int | 1 | First global step in chunk |
| end_step | int | 4 | Last global step in chunk |
| chunk_text | string | "Start by bringing... Let this cook..." | Full narrative text of chunk |
| searchable_text | string | "olive oil garlic heat" | Keywords for semantic search |
| metadata | json | `{"difficulty": "easy"}` | Additional context |

**Example Rows:**

```csv
recipe_id,title,chunk_index,start_step,end_step,chunk_text,searchable_text,metadata
4521,Spaghetti Aglio e Olio,1,1,3,"Start by bringing a large pot of salted water to a boil. While the water is heating, pour half a cup of olive oil into a large skillet and place it over low heat. Slice your six garlic cloves thinly...","boil pasta water olive oil garlic heat","{}",
4521,Spaghetti Aglio e Olio,2,4,6,"Let this cook for about two minutes until the garlic becomes fragrant. Now add the spaghetti to your boiling water...","cook garlic al dente drain","{}",
4521,Spaghetti Aglio e Olio,3,7,8,"Add the drained pasta to the skillet with the garlic oil. Toss everything together well...","combine toss serve","{}",
```

### recipes.csv

**Location:** `data/recipes.csv`

**Purpose:** Recipe metadata for search results

**Schema:**

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| recipe_id | int | 4521 | Unique ID |
| title | string | Spaghetti Aglio e Olio | Recipe name |
| cuisine | string | Italian | Cuisine type |
| difficulty | string | easy | Difficulty level |
| cook_time | int | 10 | Minutes to cook |
| prep_time | int | 5 | Minutes to prepare |
| servings | int | 4 | Number of servings |
| vegetarian | boolean | true | Dietary classification |

### food_dictionary.csv

**Location:** `data/food_dictionary.csv`

**Purpose:** Ingredient lookup by recipe

**Schema:**

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| recipe_id | int | 4521 | Recipe ID |
| recipe_name | string | Spaghetti Aglio e Olio | Recipe title |
| ingredient | string | garlic | Ingredient name |
| quantity | float | 6 | Amount |
| unit | string | cloves | Measurement unit |
| ndb_id | int | 11215 | USDA nutrition database ID |

### searchable_text_for_embeddings.csv

**Location:** `data/searchable_text_for_embeddings.csv`

**Purpose:** Text to embed for Milvus vector search

**Schema:**

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| recipe_id | int | 4521 | Recipe ID |
| searchable_text | string | "spaghetti aglio olio pasta garlic oil..." | Full text for embedding |
| embedding | float array (384) | [0.12, -0.45, ...] | Pre-computed 384-D embedding |

---

## 7. Navigator Result Schema

### NavigationResult Dataclass

**Type:** Returned by `RecipeNavigator` methods

```python
@dataclass
class NavigationResult:
    success: bool
    intent: str
    step_index: int
    chunk_index: int
    text: str
    section: str
    is_last_step: bool = False
    is_first_step: bool = False
    message: str = ""
    extra: Dict = None
```

**Example (next step):**

```python
NavigationResult(
    success=True,
    intent="nav_next",
    step_index=2,
    chunk_index=1,
    text="While the water is heating, pour half a cup of olive oil into a large skillet and place it over low heat.",
    section="steps",
    is_last_step=False,
    is_first_step=False,
    message="",
    extra={}
)
```

**Example (ingredients):**

```python
NavigationResult(
    success=True,
    intent="nav_repeat_ingredients",
    step_index=0,
    chunk_index=0,
    text="For Spaghetti Aglio e Olio you'll need: one pound of spaghetti, six garlic cloves, half a cup of olive oil, two teaspoons of salt, and a quarter teaspoon of red pepper flakes.",
    section="ingredients",
    is_last_step=False,
    is_first_step=True,
    message="",
    extra={"ingredient_count": 5}
)
```

---

## 8. TTS Input/Output Schema

### Text Input for TTS

**Type:** String

**Constraints:**
- Max length: 5000 characters (Deepgram limit)
- Typically chunks to 300-500 chars per API call

**Example:**

```
"One pound of spaghetti pasta. Six fresh garlic cloves. Half a cup of extra virgin olive oil."
```

### TTS API Request

**HTTP POST** to `https://api.deepgram.com/v1/speak`

```json
{
  "text": "One pound of spaghetti pasta...",
  "model": "aura-asteria-en",
  "encoding": "linear16",
  "container": "wav"
}
```

### TTS API Response

**Returns:** MP3 audio stream (binary)

**Saved as:** `tts_generated_speech/response_20260319_103045.mp3`

---

## 9. Audio Format Specifications

### WAV Files (Voice Recordings)

- **Format:** PCM WAV
- **Channels:** 1 (mono)
- **Sample rate:** 16,000 Hz (16 kHz)
- **Bit depth:** 16-bit signed
- **Frame size:** 512 samples (for Porcupine compatibility)
- **Location:** `voice_recordings/recording_TIMESTAMP.wav`

### MP3 Files (TTS Output)

- **Format:** MP3
- **Channels:** 1 (mono)
- **Bitrate:** 128 kbps (typical)
- **Location:** `tts_generated_speech/response_TIMESTAMP.mp3`

---

## 10. Environment Variables Schema

### Required Variables

```bash
# Deepgram API (ASR + TTS)
DEEPGRAM_API_KEY="<32-character alphanumeric token>"

# Google Gemini API (LLM)
Gemini_API_key="<long API key from Google Cloud>"

# Recipe Backend
API_BASE_URL="https://api.example.com"  # or http://localhost:5000

# Upstash Redis (Session Storage)
REDIS_URL="rediss://default:<TOKEN>@<HOST>:<PORT>"
```

### Optional Variables

```bash
PORCUPINE_ACCESS_KEY="<optional if using cloud features>"
ENVIRONMENT="development"  # or production
VAD_SILENCE_THRESHOLD=500
VAD_SILENCE_DURATION=3.0
CONFIDENCE_THRESHOLD=0.70
```

---

## 11. Error Response Schema

### API Error Response

**Type:** JSON (error case)

```json
{
  "success": false,
  "error": "Recipe not found",
  "error_code": "RECIPE_NOT_FOUND",
  "details": {
    "recipe_id": 4521,
    "reason": "ID does not exist in database"
  }
}
```

### Navigation Error Result

```python
NavigationResult(
    success=False,
    intent="nav_next",
    step_index=8,
    chunk_index=3,
    text="No more steps available. You've completed the recipe!",
    section="done",
    is_last_step=True,
    is_first_step=False,
    message="End of recipe reached",
    extra={}
)
```

---

## 12. Summary: Data Flow Schema Chain

```
User Speech
    ↓ (ASR)
Transcribed Text (plain string)
    ↓ (Intent Classifier)
Intent + Confidence + Entities
    ↓ (Dispatcher)
    ├─ Recipe Search → Retriever → Full Recipe Dict
    ├─ Navigation → Navigator → NavigationResult
    └─ QA/Small Talk → LLM → Plain String
    ↓ (LLM)
Structured JSON Response OR Plain Text
    ↓ (TTS)
MP3 Chunks (bytes)
    ↓ (Audio Player)
Playback to Speaker
    ↓ (Session Manager)
Session State → Redis/In-Memory
```

---

This schema documentation provides the formal contract for all data transformations within the BTP Voice Assistant, essential for integration, testing, and academic evaluation.
