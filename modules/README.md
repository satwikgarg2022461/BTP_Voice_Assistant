# Modules Directory Guide

Quick reference for all modules in the BTP Voice Assistant.

---

## Module Overview

| Module | File | Lines | Purpose | Dependencies |
|--------|------|-------|---------|--------------|
| **Wake Word Detection** | `wakeword.py` | ~150 | Porcupine "Hey Cook" detection | picovoice, pvrecorder |
| **Voice Activity Detection** | `vad.py` | ~95 | Record user input + silence detection | pvrecorder, numpy |
| **ASR (Deepgram)** | `deepgram_asr.py` | ~80 | Cloud ASR transcription | requests, dotenv |
| **ASR (Whisper)** | `asr.py` | ~100 | Local Whisper fallback | openai-whisper, pydub |
| **Intent Classification** | `intent_classifier.py` | ~531 | Rule + LLM intent detection | google-genai, regex |
| **Recipe Retrieval** | `retriever.py` | ~168 | Milvus vector search | pymilvus, sentence-transformers |
| **Recipe Navigator** | `navigator.py` | ~769 | Step-by-step guidance | pandas, csv |
| **LLM Response Gen** | `llm.py` | ~525 | Gemini-based responses | google-genai, json |
| **Session Manager** | `session_manager.py` | ~798 | Redis session state | upstash-redis, json |
| **Text-to-Speech** | `tts.py` | ~120 | Deepgram TTS API | requests, pydub |
| **Audio Player** | `audio_player.py` | ~200 | Chunked playback + threading | pydub, threading |

---

## Dependency Graph

```
src/main.py (main coordinator)
├── wakeword.py (background thread)
├── vad.py (record + VAD)
├── deepgram_asr.py (PRIMARY)
│   └─ asr.py (FALLBACK if Deepgram fails)
├── intent_classifier.py (rule-based + LLM)
│   └─ llm.py (LLM fallback for intent)
├── retriever.py (semantic search)
│   └─ API backend
├── navigator.py (recipe chunks + navigation)
│   └─ data/chunks.csv
│   └─ data/food_dictionary.csv
├── llm.py (response generation)
├── session_manager.py (Redis session state)
├── tts.py (audio synthesis)
└── audio_player.py (playback)
```

**Call sequence per user turn:**
```
1. wait_for_wakeword() [from wakeword.py]
2. record_once() [from vad.py]
3. transcribe() [from deepgram_asr.py OR asr.py]
4. classify() [from intent_classifier.py]
5. dispatch() based on intent:
   - SEARCH_RECIPE → search_recipes() [retriever.py]
   - NAV_* → get_next_step() etc. [navigator.py]
   - QUESTION → answer_recipe_question() [llm.py]
   - others → generate_conversational_response() [llm.py]
6. generate_recipe_response() [llm.py]
7. create_session() or update_session() [session_manager.py]
8. generate_speech() [tts.py]
9. play_chunks() [audio_player.py]
```

---

## Individual Module Summaries

### 1. wakeword.py
**Purpose:** Always-on wake word ("Hey Cook") detection  
**Key class:** `WakeWordDetector`  
**Main methods:**
- `start()`: Begin background listening
- `wait_for_detected(timeout)`: Block until wake word or timeout
- `is_detected()`: Non-blocking check

**Configuration:**
```python
WakeWordDetector(
    keyword_paths=["models/Hey-Cook_en_linux_v3_0_0.ppn"],
    sensitivities=[0.65],  # tuned for cooking noise
    device_index=-1        # auto-detect microphone
)
```

---

### 2. vad.py
**Purpose:** Record user speech with voice activity detection (silence-based endpoint detection)  
**Key class:** `ShortRecorder`  
**Main methods:**
- `rms(pcm)`: Compute RMS energy of frame
- `record_once(out_path)`: Record until 3 sec silence

**Algorithm:** Accumulates frames while speaking, stops after silence threshold crossed

**Output:** WAV file in `voice_recordings/recording_TIMESTAMP.wav`

---

### 3. deepgram_asr.py
**Purpose:** Cloud ASR via Deepgram API  
**Key class:** `DeepgramASR`  
**Main methods:**
- `transcribe(audio_path)`: Send audio → get transcription

**API:** POST to `https://api.deepgram.com/v1/listen`  
**Cost:** ~$0.0043/min  
**Latency:** 1-3 sec  
**Output:** Plain text transcription (lowercased)

---

### 4. asr.py (Fallback)
**Purpose:** Local ASR via OpenAI Whisper  
**Key class:** `WhisperASR`  
**Main methods:**
- `transcribe(audio_path)`: Load Whisper, transcribe locally

**Model:** "small" (244M params, 577 MB)  
**Latency:** ~10 sec  
**Cost:** Free (local inference)  
**Fallback to:** When Deepgram unavailable or rate-limited

---

### 5. intent_classifier.py
**Purpose:** Classify user input to Intent (17 types)  
**Key classes:**
- `Intent` (Enum): 17 intent types (NAV_NEXT, SEARCH_RECIPE, QUESTION, etc.)
- `IntentClassifier`: Hybrid rule-based + LLM classification

**Main methods:**
- `classify(text, context)`: Returns (Intent, confidence, entities)

**Algorithm:**
1. Try rule-based patterns (fast, <50ms)
2. If confidence < threshold: LLM fallback (slower, ~1-2 sec)

**Key intents:**
- Navigation: NAV_NEXT, NAV_PREV, NAV_GO_TO, NAV_REPEAT, NAV_REPEAT_INGREDIENTS
- Recipe: SEARCH_RECIPE, START_RECIPE
- Control: STOP_PAUSE, RESUME
- Other: QUESTION, SMALL_TALK, UNKNOWN

---

### 6. retriever.py
**Purpose:** Find recipes via semantic search (Milvus vector DB)  
**Key class:** `RecipeRetriever`  
**Main methods:**
- `embed_query(text)`: Generate 384-D embedding
- `search_recipes(query, top_k=5)`: Return top-K similar recipe IDs
- `fetch_full_recipe_details(recipe_id)`: Get ingredients + instructions from API

**Embedding model:** SentenceTransformer all-MiniLM-L6-v2 (384-D)  
**Backend DB:** Milvus (recipes_demo.db)  
**Similarity metric:** Cosine distance  
**Latency:** 100-500 ms total

---

### 7. navigator.py
**Purpose:** Step-by-step recipe navigation + state tracking  
**Key classes:**
- `RecipeData`: Recipe metadata + chunks
- `NavigationResult`: Return type for navigation operations
- `RecipeNavigator`: Main API

**Main methods:**
- `load_recipe(recipe_id)`: Load chunks from CSV
- `get_next_step()`, `get_previous_step()`, `jump_to_step()`: Navigate
- `get_current_ingredients()`: Show all ingredients
- `restart()`: Go back to ingredients

**Data sources:**
- `data/chunks.csv`: Recipe chunks (start_step, end_step, text)
- `data/food_dictionary.csv`: Ingredients by recipe

**State tracked:**
- `current_step` (1-based, 0 = ingredients)
- `current_chunk` (1-based)

---

### 8. llm.py
**Purpose:** Generate structured responses + answer cooking questions  
**Key class:** `RecipeLLM`  
**Main methods:**
- `generate_recipe_response(query, recipe_results)`: Returns structured JSON
  ```json
  {
    "greeting": "...",
    "ingredients": [{"text": "...", "spoken": false}, ...],
    "steps": [{"step_num": 1, "text": "...", "spoken": false}, ...],
    "closing": "..."
  }
  ```
- `generate_conversational_response()`: Plain text for non-recipe intents
- `answer_recipe_question()`: Answer "Can I use X instead of Y?" type questions

**LLM:** Google Gemini 2.5 Flash  
**Latency:** 1-3 sec  
**Cost:** ~$0.075/1M input tokens (negligible for prototype)

---

### 9. session_manager.py
**Purpose:** Manage recipe session state (Redis-backed)  
**Key class:** `SessionManager`  
**Main methods:**
- `create_session()`: Start new recipe session
- `get_session()`, `update_session()`: Read/write session
- `add_to_history()`, `get_history()`: Conversation tracking

**Backend:**
- Primary: Upstash Redis (cloud, persistent)
- Fallback: In-memory dict (if Redis unavailable)

**Session fields:**
- `recipe_id`, `recipe_title`, `current_step`, `current_chunk`
- `response_structure` (from LLM)
- `ingredients_spoken`, `steps_spoken` (progress tracking)
- `conversation_history` (last 10 turns)

**TTL:** 3600 seconds (configurable)

---

### 10. tts.py
**Purpose:** Text-to-speech via Deepgram API  
**Key class:** `RecipeTTS`  
**Main methods:**
- `generate_speech(text)`: Text → MP3 file
- `split_into_chunks(text)`: Split long text into TTS-friendly chunks

**API:** POST to `https://api.deepgram.com/v1/speak`  
**Voice model:** aura-asteria-en (natural, friendly)  
**Cost:** ~$0.003/1K characters  
**Latency:** 1-5 sec per text chunk

---

### 11. audio_player.py
**Purpose:** Play MP3 chunks with concurrent wake word detection  
**Key classes:**
- `AudioPlayer`: Simple sequential playback
- `ChunkedAudioPlayer`: Concurrent playback + pause/resume

**Main methods:**
- `play_chunks(mp3_chunks)`: Stream MP3 byte chunks (non-blocking)
- `pause()`, `resume()`, `stop()`: Playback control

**Architecture:** Background playback thread + main thread listening for wake word

---

## Data Files Dependencies

| File | Used by | Format |
|------|---------|--------|
| `data/chunks.csv` | navigator.py | Recipe chunks (start_step, end_step, text) |
| `data/recipes.csv` | navigator.py | Recipe metadata |
| `data/food_dictionary.csv` | navigator.py | Ingredients by recipe_id |
| `data/searchable_text_for_embeddings.csv` | retriever.py | Pre-computed embeddings |
| `models/Hey-Cook_en_linux_v3_0_0.ppn` | wakeword.py | Porcupine wake word model |
| `voice_recordings/` | Output | WAV recordings from ShortRecorder |
| `ASR_text/` | Output | Transcriptions from ASR |
| `tts_generated_speech/` | Output | MP3 files from RecipeTTS |

---

## Environment Dependencies

### Required Packages

```bash
# Audio I/O
pyaudio
pysoundfile

# Wake Word
picovoice
pvrecorder

# ASR
requests  # For Deepgram API
openai-whisper  # For local fallback

# Intent Classification
google-genai  # For Gemini

# Retrieval
pymilvus  # Vector DB
sentence-transformers

# LLM
google-genai  # Gemini

# Session
upstash-redis  # Cloud session store

# Utils
pandas
python-dotenv
```

### API Keys (Environment Variables)

```bash
DEEPGRAM_API_KEY="..."           # ASR + TTS
Gemini_API_key="..."             # Intent + LLM
REDIS_URL="..."                  # Session storage (optional)
API_BASE_URL="http://localhost:5000"  # Recipe backend
```

---

## Extension Points

### Adding a New Intent Type

1. Add to `Intent` enum in `intent_classifier.py`
   ```python
   class Intent(Enum):
       NEW_INTENT = "new_intent"
   ```

2. Add pattern rules
   ```python
   Intent.NEW_INTENT: [
       {"pattern": r"...", "confidence": 0.90},
   ]
   ```

3. Add handler in `src/main.py` dispatch
   ```python
   if intent == Intent.NEW_INTENT:
       result = handle_new_intent(...)
   ```

### Adding a New ASR Provider

1. Create `asr_provider.py` with `Provider.transcribe()` method
2. Import in `src/main.py`
3. Add to fallback chain

---

## Testing

See [tests/](../tests/) directory for unit tests for individual modules.

```bash
pytest tests/test_intent_classifier.py
pytest tests/test_navigator.py
pytest tests/test_llm.py
```

---

This modular architecture enables easy testing, debugging, and extension of individual components without affecting the entire system.
