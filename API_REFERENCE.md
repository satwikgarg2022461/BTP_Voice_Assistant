# API Reference Documentation

This document specifies the interface, inputs, outputs, and usage patterns for each core module in the BTP Voice Assistant.

---

## `modules/wakeword.py` – Wake Word Detection

### Class: `WakeWordDetector`

Manages always-on wake word detection using Picovoice Porcupine.

#### Constructor

```python
WakeWordDetector(
    keyword_paths: List[str],
    model_path: Optional[str] = None,
    sensitivities: Optional[List[float]] = None,
    access_key: Optional[str] = None,
    device_index: int = -1
)
```

**Parameters:**
- `keyword_paths` (List[str]): Paths to .ppn model files (e.g., ["models/Hey-Cook_en_linux_v3_0_0.ppn"])
- `model_path` (str, optional): Path to Porcupine model (auto-detected if None)
- `sensitivities` (List[float], optional): Sensitivity per keyword (0.0-1.0, default 0.5)
- `access_key` (str, optional): Picovoice access key (required for cloud-connected features)
- `device_index` (int): Audio device index (-1 = default)

**Raises:**
- `ValueError`: If keyword_paths empty or models not found

#### Methods

##### `start()`
Starts the wake word detection thread.

```python
def start() -> None
```

**Behavior:**
- Launches background daemon thread
- Thread runs PvRecorder continuously
- Non-blocking; returns immediately

---

##### `is_detected()`
Checks if wake word was detected without blocking.

```python
def is_detected() -> bool
```

**Returns:** `True` if detected since last call, `False` otherwise
**Side effect:** Resets internal flag after returning `True`

---

##### `wait_for_detected(timeout: Optional[float] = None)`
Blocks until wake word detected or timeout.

```python
def wait_for_detected(timeout: Optional[float] = None) -> bool
```

**Parameters:**
- `timeout` (float, optional): Max seconds to wait (None = infinite)

**Returns:** `True` if detected, `False` if timeout

---

##### `stop()`
Stops the detection thread and releases resources.

```python
def stop() -> None
```

---

## `modules/vad.py` – Voice Activity Detection & Recording

### Class: `ShortRecorder`

Captures audio with voice activity detection (VAD) using silence-based endpoint detection.

#### Constructor

```python
ShortRecorder(
    sample_rate: int = 16000,
    frame_length: int = 512,
    pre_roll_secs: float = 1.0,
    silence_thresh: int = 500,
    silence_duration: float = 3.0
)
```

**Parameters:**
- `sample_rate` (int): Sample rate in Hz (16000 for ASR)
- `frame_length` (int): Samples per frame (512 for Porcupine compatibility)
- `pre_roll_secs` (float): Seconds of pre-trigger audio to capture
- `silence_thresh` (int): RMS energy threshold for silence (0-32767)
- `silence_duration` (float): Seconds of silence before stopping recording

#### Methods

##### `rms(pcm: List[int]) -> float`
Computes root mean square energy of audio frame.

```python
def rms(pcm: List[int]) -> float
```

**Parameters:**
- `pcm` (List[int]): Audio frame (PCM samples, 16-bit signed)

**Returns:** RMS energy value (0-32767 range)

---

##### `record_once(out_path: str = "temp.wav") -> str`
Records one utterance with silence detection.

```python
def record_once(out_path: str = "temp.wav") -> str
```

**Parameters:**
- `out_path` (str): Output WAV file path

**Returns:** Path to saved WAV file

**Behavior:**
1. Fills pre-roll buffer (1 second default)
2. Starts recording
3. Accumulates frames while speaking
4. Detects silence (RMS < threshold for 3 seconds)
5. Stops, combines pre-roll + recorded audio
6. Saves as 16-bit WAV

**Raises:**
- `IOError`: If cannot write to out_path

---

## `modules/intent_classifier.py` – Intent Classification

### Enum: `Intent`

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

### Class: `IntentClassifier`

Hybrid rule-based + LLM intent classification.

#### Constructor

```python
IntentClassifier(
    confidence_threshold: float = 0.7,
    use_llm_fallback: bool = True
)
```

**Parameters:**
- `confidence_threshold` (float): Min confidence (0-1) for accepting rule-based classification
- `use_llm_fallback` (bool): Whether to use Gemini LLM if confidence < threshold

---

#### Methods

##### `classify(text: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Intent, float, Dict[str, Any]]`

Classifies user input text to an Intent.

```python
def classify(text: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Intent, float, Dict[str, Any]]
```

**Parameters:**
- `text` (str): User transcription
- `context` (Dict, optional): Context dict with keys:
  - `current_intent` (Intent): Previous intent (for context boost)
  - `recipe_loaded` (bool): Whether recipe is active
  - `conversation_history` (List[str]): Recent user inputs

**Returns:** Tuple of:
- `intent` (Intent): Classified intent
- `confidence` (float): 0-1 confidence score
- `entities` (Dict): Extracted entities, e.g., `{"target_step": 5, "ingredient": "garlic"}`

**Classification Flow:**
1. Test rule patterns (regex) against text
2. If confidence ≥ threshold: return rule result
3. Else if use_llm_fallback: query Gemini for classification
4. Else: return UNKNOWN

**Example:**
```python
classifier = IntentClassifier()
intent, conf, entities = classifier.classify("show me the next step")
# Returns: (Intent.NAV_NEXT, 0.95, {})

intent, conf, entities = classifier.classify("how much salt should i add?")
# Returns: (Intent.QUESTION, 0.90, {"ingredient": "salt"})
```

---

## `modules/retriever.py` – Recipe Retrieval

### Class: `RecipeRetriever`

Retrieves recipes via semantic search using Milvus embeddings.

#### Constructor

```python
RecipeRetriever(
    db_path: str = "recipes_demo.db",
    collection_name: str = "recipes_collection",
    model_name: str = 'all-MiniLM-L6-v2'
)
```

**Parameters:**
- `db_path` (str): Milvus database file path
- `collection_name` (str): Milvus collection name
- `model_name` (str): SentenceTransformer model name

**Raises:**
- `Exception`: If Milvus collection does not exist

---

#### Methods

##### `embed_query(query_text: str) -> np.ndarray`

Generates 384-D embedding for query text.

```python
def embed_query(query_text: str) -> np.ndarray
```

**Parameters:**
- `query_text` (str): Text to embed (e.g., "pasta with garlic")

**Returns:** Normalized 384-D float32 numpy array

---

##### `search_recipes(query_text: str, top_k: int = 5) -> List[Dict]`

Semantic search over recipe embeddings.

```python
def search_recipes(query_text: str, top_k: int = 5) -> List[Dict]
```

**Parameters:**
- `query_text` (str): Search query
- `top_k` (int): Number of results to return

**Returns:** List of recipe dicts, each with:
```python
{
    "recipe_id": int,
    "title": str,
    "similarity_score": float,  # 0-1, higher = better match
    "metadata": Dict
}
```

---

##### `fetch_full_recipe_details(recipe_id: int) -> Optional[Dict]`

Fetches complete recipe data from backend API.

```python
def fetch_full_recipe_details(recipe_id: int) -> Optional[Dict]
```

**Parameters:**
- `recipe_id` (int): Recipe ID from search results

**Returns:** Recipe dict or None if not found:
```python
{
    "recipe_id": int,
    "title": str,
    "ingredients": [
        {
            "ingredient": str,
            "quantity": float,
            "unit": str,
            "state": str,
            "ndb_id": int
        },
        ...
    ],
    "instructions": [
        {"step_number": int, "step": str},
        ...
    ],
    "cuisine": str,
    "cook_time": int,  # minutes
    "prep_time": int   # minutes
}
```

**Raises:**
- `requests.RequestException`: If API call fails

---

## `modules/navigator.py` – Recipe Navigation

### Dataclass: `RecipeData`

```python
@dataclass
class RecipeData:
    recipe_id: int
    title: str
    chunks: List[ChunkData]
    all_ingredients: List[str]
    total_steps: int
    total_chunks: int
```

### Dataclass: `NavigationResult`

Return type for all navigation operations.

```python
@dataclass
class NavigationResult:
    success: bool                 # Operation succeeded
    intent: str                   # The intent that triggered this
    step_index: int              # 1-based global step (0 = ingredients)
    chunk_index: int             # 1-based chunk number
    text: str                    # Text to speak/display
    section: str                 # "ingredients" | "steps" | "done"
    is_last_step: bool = False
    is_first_step: bool = False
    message: str = ""            # Status message
    extra: Dict = None           # Optional extra data
```

### Class: `RecipeNavigator`

Manages recipe navigation from CSV data.

#### Constructor

```python
RecipeNavigator(
    chunks_csv: str = "data/chunks.csv",
    recipes_csv: str = "data/recipes.csv",
    food_dict_csv: str = "data/food_dictionary.csv"
)
```

---

#### Methods

##### `load_recipe(recipe_id: int) -> Optional[RecipeData]`

Loads recipe chunks and ingredients from CSV.

```python
def load_recipe(recipe_id: int) -> Optional[RecipeData]
```

**Returns:** RecipeData object or None if not found

---

##### `get_current_step(session: Dict) -> NavigationResult`

Returns current step text.

```python
def get_current_step(session: Dict) -> NavigationResult
```

**Session fields used:**
- `step_index` (int): Current global step
- `chunk_index` (int): Current chunk
- `recipe_id` (int): Recipe ID

---

##### `get_next_step(session: Dict) -> NavigationResult`

Advances to next step.

```python
def get_next_step(session: Dict) -> NavigationResult
```

**Updates session:** Increments `step_index` and `chunk_index` if needed

---

##### `get_previous_step(session: Dict) -> NavigationResult`

Goes to previous step.

```python
def get_previous_step(session: Dict) -> NavigationResult
```

---

##### `jump_to_step(session: Dict, target_step: int) -> NavigationResult`

Jumps to specific step number.

```python
def jump_to_step(session: Dict, target_step: int) -> NavigationResult
```

**Parameters:**
- `target_step` (int): Target step number (1-based, 0 = ingredients)

---

##### `get_current_ingredients(session: Dict) -> NavigationResult`

Returns all ingredients for current recipe.

```python
def get_current_ingredients(session: Dict) -> NavigationResult
```

---

##### `restart(session: Dict) -> NavigationResult`

Resets to beginning (ingredients).

```python
def restart(session: Dict) -> NavigationResult
```

---

## `modules/llm.py` – LLM Response Generation

### Class: `RecipeLLM`

Generates structured responses using Google Gemini.

#### Constructor

```python
RecipeLLM(model_name: str = "gemini-2.5-flash")
```

**Raises:**
- `ValueError`: If Gemini_API_key not in environment

---

#### Methods

##### `generate_recipe_response(user_query: str, recipe_results: List[Dict], return_json: bool = True) -> Union[Dict, str]`

Generates structured JSON response for recipe display.

```python
def generate_recipe_response(
    user_query: str,
    recipe_results: List[Dict],
    return_json: bool = True,
    conversation_history: Optional[List[Dict]] = None
) -> Union[Dict, str]
```

**Parameters:**
- `user_query` (str): Original user query
- `recipe_results` (List[Dict]): Results from RecipeRetriever
- `return_json` (bool): If True, return structured JSON; else plain text
- `conversation_history` (List[Dict], optional): Previous turns

**Returns (if return_json=True):** 
```python
{
    "greeting": str,           # "Let me show you how to make..."
    "ingredients": [
        {
            "text": str,       # "1 cup flour"
            "spoken": bool     # False initially
        },
        ...
    ],
    "steps": [
        {
            "step_num": int,   # 1-based
            "text": str,       # "Mix ingredients..."
            "spoken": bool     # False initially
        },
        ...
    ],
    "closing": str            # "Enjoy your meal!"
}
```

**Returns (if return_json=False):** Plain text concatenation of all sections

**Fallback behavior:**
1. Try LLM generation with structured prompt
2. If invalid JSON returned: extract from markdown code blocks
3. If still invalid: manually construct from recipe_results
4. If all fails: return error message

---

##### `generate_conversational_response(user_query: str, current_recipe: Optional[str] = None, session_context: Optional[Dict] = None, conversation_history: Optional[List[Dict]] = None) -> str`

Generates natural language response for non-recipe intents.

```python
def generate_conversational_response(
    user_query: str,
    current_recipe: Optional[str] = None,
    session_context: Optional[Dict] = None,
    conversation_history: Optional[List[Dict]] = None
) -> str
```

**Parameters:**
- `user_query` (str): User's input
- `current_recipe` (str, optional): Recipe title if one is active
- `session_context` (Dict, optional): Current state (recipe, step, etc.)
- `conversation_history` (List[Dict], optional): Last 6 turns

**Returns:** 1-3 sentence response suitable for TTS

**Example:**
```python
llm = RecipeLLM()
response = llm.generate_conversational_response(
    user_query="what's garlic?",
    current_recipe="Spaghetti Aglio e Olio",
    conversation_history=[...]
)
# Returns: "Garlic is a bulbous plant used in cooking. It adds a strong, pungent flavor..."
```

---

##### `answer_recipe_question(question: str, recipe_context: str, conversation_history: Optional[List[Dict]] = None) -> str`

Answers questions about recipes using RAG.

```python
def answer_recipe_question(
    question: str,
    recipe_context: str,
    conversation_history: Optional[List[Dict]] = None
) -> str
```

**Parameters:**
- `question` (str): User's question (e.g., "Can I use salt instead?")
- `recipe_context` (str): Current recipe title + instructions
- `conversation_history` (List[Dict], optional): Conversation history

**Returns:** 2-4 sentence answer

---

## `modules/session_manager.py` – Session State

### Class: `SessionManager`

Manages session state via Upstash Redis (with in-memory fallback).

#### Constructor

```python
SessionManager(session_ttl: int = 3600)
```

**Parameters:**
- `session_ttl` (int): Session time-to-live in seconds

**Backend:**
- Primary: Upstash Redis (rediss://...) if REDIS_URL configured
- Fallback: In-memory dict if Redis unavailable

---

#### Methods

##### `create_session(session_id: str, recipe_id: str, recipe_title: str, total_steps: int, recipe_data: Optional[Dict] = None) -> Dict`

Creates new recipe session.

```python
def create_session(
    session_id: str,
    recipe_id: str,
    recipe_title: str,
    total_steps: int,
    recipe_data: Optional[Dict] = None
) -> Dict
```

**Returns:** Session dict:
```python
{
    "session_id": str,
    "recipe_id": str,
    "recipe_title": str,
    "current_step": 1,
    "current_chunk": 1,
    "total_steps": int,
    "created_at": str,  # ISO timestamp
    "last_updated": str,
    "is_paused": false,
    "response_structure": {...},  # From LLM
    "conversation_history": []
}
```

---

##### `get_session(session_id: str) -> Optional[Dict]`

Retrieves session from storage.

```python
def get_session(session_id: str) -> Optional[Dict]
```

---

##### `update_session(session_id: str, updates: Dict) -> bool`

Updates session fields.

```python
def update_session(session_id: str, updates: Dict) -> bool
```

**Parameters:**
- `updates` (Dict): Fields to update (e.g., `{"current_step": 5}`)

**Returns:** True if successful

---

##### `add_to_history(session_id: str, role: str, content: str) -> bool`

Appends message to conversation history.

```python
def add_to_history(session_id: str, role: str, content: str) -> bool
```

**Parameters:**
- `role` (str): "user" or "assistant"
- `content` (str): Message text

**Returns:** True if successful

---

##### `get_history(session_id: str, limit: int = 10) -> List[Dict]`

Retrieves conversation history.

```python
def get_history(session_id: str, limit: int = 10) -> List[Dict]
```

**Returns:** List of dicts: `[{"role": "user"/"assistant", "content": str}, ...]`

---

## `modules/tts.py` – Text-to-Speech

### Class: `RecipeTTS`

Generates speech chunks via Deepgram TTS API.

#### Constructor

```python
RecipeTTS(
    model: str = "aura-asteria-en",
    output_dir: str = "tts_generated_speech"
)
```

**Parameters:**
- `model` (str): Deepgram voice model
- `output_dir` (str): Directory for MP3 output

---

#### Methods

##### `generate_speech(text: str, output_file: Optional[str] = None) -> str`

Generates speech audio for text.

```python
def generate_speech(text: str, output_file: Optional[str] = None) -> str
```

**Parameters:**
- `text` (str): Text to synthesize
- `output_file` (str, optional): Output MP3 path (auto-generated if None)

**Returns:** Path to generated MP3 file

**API Call:**
- Endpoint: `https://api.deepgram.com/v1/speak?model=aura-asteria-en`
- Auth: Bearer token (DEEPGRAM_API_KEY)

---

##### `split_into_chunks(text: str, max_chunk_length: int = 300) -> List[str]`

Splits long text into TTS-friendly chunks.

```python
def split_into_chunks(text: str, max_chunk_length: int = 300) -> List[str]
```

**Parameters:**
- `text` (str): Text to split
- `max_chunk_length` (int): Max chars per chunk

**Returns:** List of chunks, split by sentences

---

## `modules/audio_player.py` – Audio Playback

### Class: `ChunkedAudioPlayer`

Plays MP3 chunks with concurrent recording support.

#### Constructor

```python
ChunkedAudioPlayer(device_index: int = -1)
```

---

#### Methods

##### `play_chunks(mp3_chunks: List[bytes]) -> None`

Plays list of MP3 byte chunks.

```python
def play_chunks(mp3_chunks: List[bytes]) -> None
```

**Behavior:**
- Streams chunks concurrently
- Non-blocking; returns after starting playback
- Background thread handles actual audio output

---

##### `pause() -> None`

Pauses current playback.

```python
def pause() -> None
```

---

##### `resume() -> None`

Resumes paused playback.

```python
def resume() -> None
```

---

##### `stop() -> None`

Stops playback and cleanup.

```python
def stop() -> None
```

---

## Usage Workflow Example

```python
# 1. Initialize components
wake = WakeWordDetector(["models/Hey-Cook_en_linux_v3_0_0.ppn"])
recorder = ShortRecorder()
classifier = IntentClassifier()
retriever = RecipeRetriever()
navigator = RecipeNavigator()
llm = RecipeLLM()
session_mgr = SessionManager()
tts = RecipeTTS()
player = ChunkedAudioPlayer()

# 2. Wait for wake word
wake.start()
wake.wait_for_detected()

# 3. Record user input
audio_path = recorder.record_once()

# 4. Classify intent
intent, conf, entities = classifier.classify(
    "How do I make pasta?"
)

# 5a. If SEARCH_RECIPE
if intent == Intent.SEARCH_RECIPE:
    results = retriever.search_recipes("pasta", top_k=5)
    full_recipe = retriever.fetch_full_recipe_details(
        results[0]["recipe_id"]
    )
    
    response = llm.generate_recipe_response(
        user_query="How do I make pasta?",
        recipe_results=[full_recipe],
        return_json=True
    )
    
    session = session_mgr.create_session(
        session_id="user_001",
        recipe_id=str(full_recipe["recipe_id"]),
        recipe_title=full_recipe["title"],
        total_steps=len(full_recipe["instructions"])
    )

# 5b. If NAVIGATION
if intent == Intent.NAV_NEXT:
    result = navigator.get_next_step(session)
    session_mgr.update_session(
        "user_001",
        {"current_step": result.step_index}
    )
    response = result.text

# 6. Generate speech
mp3_file = tts.generate_speech(response)

# 7. Play
player.play_chunks([open(mp3_file, 'rb').read()])

# 8. Go back to step 2
```

---

## Error Codes & Exceptions

| Exception | Module | Cause | Handling |
|-----------|--------|-------|----------|
| `ValueError` | WakeWordDetector | Invalid keyword path | Check file exists |
| `IOError` | ShortRecorder | Cannot write WAV | Check permissions |
| `requests.RequestException` | RecipeRetriever | API unavailable | Use cached data |
| `Exception` | RecipeRetriever | Milvus unavailable | Fall back to search API |
| `ValueError` | RecipeLLM | Gemini API key missing | Check env vars |
| `redis.exceptions.ConnectionError` | SessionManager | Redis unreachable | Use in-memory fallback |

---

This API reference provides all necessary information for extending, integrating, or debugging the BTP Voice Assistant.
