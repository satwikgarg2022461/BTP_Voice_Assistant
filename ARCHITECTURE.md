# System Architecture

## Overview

The BTP Voice Assistant is a **conversational cooking assistant** that enables hands-free recipe access through speech. The system listens for a wake word, processes natural language commands, retrieves recipes, and provides step-by-step guidance via speech synthesis.

This document describes the system architecture, component interactions, and data flow for academic understanding and system evaluation.

---

## High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION LOOP                             │
└─────────────────────────────────────────────────────────────────────────┘

Background Thread (Daemon)          →    Main Application Thread
─────────────────────────────────        ──────────────────────
┌──────────────────────────────┐         ┌──────────────────────────┐
│  Wake Word Detection         │         │  1. Wait for Wake Word   │
│  (Porcupine - "Hey Cook")    │         │  2. Pause Active Player  │
│  • Always running            │────────→│  3. Record User Input    │
│  • PvRecorder → PCM frames   │         │     (VAD + Timeout)      │
│  • Triggers detected_event   │         │  4. Transcribe (ASR)     │
└──────────────────────────────┘         │  5. Classify Intent      │
                                         │  6. Dispatch to Handler  │
                                         │     └─ Recipe Search     │
                                         │     └─ Navigation       │
                                         │     └─ QA              │
                                         │     └─ Small Talk      │
                                         │  7. Generate Response    │
                                         │  8. TTS → Chunked Play   │
                                         └──────────────────────────┘
                                                     ↓
                                         ┌──────────────────────────┐
                                         │  Resume Previous Audio   │
                                         │  (if no valid command)   │
                                         └──────────────────────────┘
```

---

## Component Architecture

```
PIPELINE STAGES
===============

 1. AUDIO CAPTURE
    ├─ WakeWordDetector (modules/wakeword.py)
    │  └─ Porcupine model: Hey-Cook_en_linux_v3_0_0.ppn
    │
    └─ ShortRecorder (modules/vad.py)
       ├─ PvRecorder (hardware interface)
       ├─ Voice Activity Detection (silence threshold)
       └─ Saves: voice_recordings/recording_TIMESTAMP.wav

 2. TRANSCRIPTION
    ├─ DeepgramASR (modules/deepgram_asr.py) [PRODUCTION]
    │  └─ API call to Deepgram v1/listen
    │
    └─ WhisperASR (modules/asr.py) [LOCAL FALLBACK]
       └─ OpenAI Whisper local model
    
    Output: ASR_text/recording_TIMESTAMP.txt

 3. INTENT CLASSIFICATION
    └─ IntentClassifier (modules/intent_classifier.py)
       ├─ Rule-based patterns (17 intent types)
       ├─ LLM fallback (Gemini 2.5 Flash) if confidence < threshold
       └─ Output: (Intent, confidence_score, extracted_entities)

 4. REQUEST DISPATCH (based on Intent)
    │
    ├─ SEARCH_RECIPE intent
    │  └─ RecipeRetriever (modules/retriever.py)
    │     ├─ Milvus vector search (recipes_demo.db)
    │     ├─ SentenceTransformer embeddings (all-MiniLM-L6-v2)
    │     └─ API call for full recipe details
    │
    ├─ NAVIGATION intents (NAV_NEXT, NAV_PREV, NAV_GO_TO, etc.)
    │  └─ RecipeNavigator (modules/navigator.py)
    │     ├─ CSV-based chunks (data/chunks.csv)
    │     ├─ Ingredients lookup (data/food_dictionary.csv)
    │     └─ State tracking (chunk_index, step_index)
    │
    ├─ QUESTION intent
    │  └─ RecipeLLM.answer_recipe_question() (modules/llm.py)
    │     └─ Gemini API with RAG context
    │
    └─ SMALL_TALK + others
       └─ RecipeLLM.generate_conversational_response() (modules/llm.py)

 5. RESPONSE GENERATION
    └─ RecipeLLM (modules/llm.py)
       ├─ For recipes: Structured JSON
       │  ├─ greeting
       │  ├─ ingredients (with "spoken" tracking)
       │  ├─ steps (with "spoken" tracking)
       │  └─ closing
       │
       └─ For other intents: Plain text response

 6. SESSION STATE MANAGEMENT
    └─ SessionManager (modules/session_manager.py)
       ├─ Backend: Upstash Redis (cloud) or in-memory fallback
       ├─ Stores:
       │  ├─ Recipe metadata (ID, title, total steps)
       │  ├─ Current navigation position
       │  ├─ Ingredients/steps spoken tracker
       │  ├─ Conversation history
       │  └─ Session TTL (3600 seconds)
       │
       └─ Enables: resume after pause/interruption

 7. TEXT-TO-SPEECH
    └─ RecipeTTS (modules/tts.py)
       ├─ Deepgram TTS API v1/speak
       ├─ Chunks long text into sentences (optimal TTS size)
       └─ Returns MP3 chunks

 8. AUDIO PLAYBACK
    ├─ ChunkedAudioPlayer (modules/audio_player.py) [PRIMARY]
    │  ├─ Concurrent playback + recording thread
    │  ├─ Pause/resume capabilities
    │  └─ Chunk streaming from TTS
    │
    └─ AudioPlayer (modules/audio_player.py) [SIMPLE]
       └─ Sequential playback
```

---

## Threading Model

The system uses **dual-thread architecture**:

### Thread 1: Wake Word Detection (Daemon, Background)
```
while True:
    frame = pvrecorder.read()          # 512 samples @ 16 kHz
    process_frame(porcupine, frame)
    if wake_word_detected:
        detected_event.set()           # Signal main thread
```

**Properties:**
- Runs continuously, even during TTS playback
- Never blocks or exits
- Wakes up main thread on detection
- Uses negligible CPU (frame-by-frame processing)

### Thread 2: Main Application Loop (Foreground)
```
while True:
    detected_event.wait()              # Block until wake word
    pause_audio_player()               # Pause current playback
    detected_event.clear()
    
    recording = record_user_input()    # With VAD timeout
    if no_input or empty:
        resume_audio_player()
        continue
    
    transcription = asr(recording)
    intent = classify_intent(transcription)
    response = dispatch(intent)        # Retrieve, navigate, or generate
    
    play_response(response)            # May be interrupted by wake word
```

---

## Data Flow Example: "How do I make pasta?"

```
Step 1: User says "Hey Cook" (wake word)
   └─ WakeWordDetector detects → detected_event.set()

Step 2: Main thread wakes up, pauses current playback, prompts for input

Step 3: User says "How do I make pasta?"
   └─ ShortRecorder captures audio
   └─ DeepgramASR transcribes → "how do i make pasta"
   └─ Saves to ASR_text/recording_20260319_XXXXXX.txt

Step 4: IntentClassifier analyzes
   ├─ Checks rules: matches "how do (make|cook|prepare)" pattern
   ├─ Confidence: HIGH (0.90)
   ├─ Intent: SEARCH_RECIPE
   └─ Threshold check: 0.90 > 0.70 (no LLM fallback needed)

Step 5: RecipeRetriever (intent handler)
   ├─ Embed query: "how do i make pasta" → 384-D vector
   ├─ Milvus search: top_k=5 similar recipe vectors
   ├─ Fetch full recipe via API: recipe_id=4521 (Spaghetti Aglio e Olio)
   └─ Return: recipe metadata with ingredients + instructions

Step 6: RecipeLLM generates structured response
   ├─ Prompt: LLM instruction + recipe details
   ├─ LLM generates JSON:
   │  {
   │    "greeting": "Let me show you how to make Spaghetti Aglio e Olio...",
   │    "ingredients": [
   │      {"text": "1 pound spaghetti", "spoken": false},
   │      {"text": "6 cloves garlic", "spoken": false},
   │      ...
   │    ],
   │    "steps": [
   │      {"step_num": 1, "text": "Bring a large pot of salted water...", "spoken": false},
   │      ...
   │    ],
   │    "closing": "Enjoy your freshly made pasta..."
   │  }
   └─ Validates structure, extracts from markdown if needed

Step 7: SessionManager persists session
   ├─ Redis key: session:user_001
   ├─ Stores: recipe_id=4521, current_step=0, response_structure
   └─ TTL: 3600 seconds

Step 8: RecipeTTS generates speech
   ├─ Chunks: greeting + ingredients + steps + closing (split by sentences)
   ├─ Deepgram API: text→speech (MP3)
   └─ Returns: list of MP3 byte chunks

Step 9: ChunkedAudioPlayer plays
   ├─ Streams chunks concurrently
   ├─ Background thread continues listening for wake word
   └─ If interrupted: pauses, waits for next command

Step 10: If user interrupts with "next"
   └─ Cycle repeats from step 2 (wake word detected again)
```

---

## Data Storage

### Persistent Storage

| Location | Format | Purpose | Size |
|----------|--------|---------|------|
| `data/chunks.csv` | CSV | Recipe chunks indexed by recipe_id, chunk_index, step_num | ~50 MB |
| `data/recipes.csv` | CSV | Recipe metadata (ID, title, cuisine, etc.) | ~2 MB |
| `data/food_dictionary.csv` | CSV | Ingredients by recipe_id | ~10 MB |
| `data/searchable_text_for_embeddings.csv` | CSV | Pre-computed searchable text for each recipe | ~5 MB |
| `models/Hey-Cook_en_linux_v3_0_0.ppn` | Binary | Porcupine wake word model (pre-trained) | ~2 MB |
| `recipes_demo.db` | Milvus vector DB | 384-D embeddings for semantic search | ~100 MB |

### Runtime Output

| Directory | Purpose | Cleanup |
|-----------|---------|---------|
| `voice_recordings/` | Raw WAV files from microphone | Manual (can grow large) |
| `ASR_text/` | Transcribed text from each recording | Manual or by date |
| `tts_generated_speech/` | MP3 chunks from Deepgram TTS | Manual or by date |

### Session Storage (Redis)

- **Backend**: Upstash Redis (cloud, JSON over HTTP)
- **Fallback**: In-memory dictionary (if Redis unavailable)
- **Keys**: `session:{session_id}`, `conversation_history:{session_id}`
- **TTL**: 3600 seconds (1 hour)
- **Format**: JSON serialized

---

## External Dependencies

### APIs

1. **Porcupine (Wake Word)**
   - Provider: Picovoice
   - Model: Hey-Cook_en_linux_v3_0_0.ppn
   - Usage: Always-on wake word detection
   - Cost: One-time (local model)

2. **Deepgram (ASR - Primary)**
   - Endpoint: `https://api.deepgram.com/v1/listen`
   - Auth: Bearer token (DEEPGRAM_API_KEY)
   - Input: WAV audio
   - Output: JSON transcription
   - Cost: ~$0.0043 per minute

3. **Deepgram (TTS)**
   - Endpoint: `https://api.deepgram.com/v1/speak`
   - Auth: Bearer token (DEEPGRAM_API_KEY)
   - Input: Text + voice model
   - Output: MP3 stream
   - Cost: ~$0.003 per 1K characters

4. **Google Gemini (LLM - Intent Fallback + Responses)**
   - Endpoint: Google AI API
   - Auth: API key (Gemini_API_key)
   - Models: `gemini-2.5-flash` (fast), fallback to `gemini-1.5-flash`
   - Cost: Depends on input/output tokens (~$0.075 per 1M input, $0.30 per 1M output)

5. **Recipe API (Custom Backend)**
   - Endpoint: `{API_BASE_URL}/search-recipe/{recipe_id}`
   - Auth: None (internal)
   - Output: Recipe JSON with ingredients + instructions
   - Cost: Internal (your server)

6. **Upstash Redis (Session Storage)**
   - Provider: Upstash
   - Endpoint: Configured via REDIS_URL
   - Auth: Token-based
   - Cost: Pay-as-you-go (~$0.20 per 100k commands)

### Local Dependencies

- **Python 3.10.18**
- **PyAudio + PortAudio** (audio I/O)
- **pysoundfile / SoundFile** (WAV handling)
- **Milvus** (vector search)
- **SentenceTransformer** (embeddings)
- **Google GenAI SDK** (LLM access)
- **Upstash Redis SDK** (session management)

---

## Configuration & Tuning

### Key Parameters

| Parameter | Location | Default | Purpose |
|-----------|----------|---------|---------|
| `wake_sensitivity` | src/main.py | 0.65 | Porcupine sensitivity (0.0-1.0) |
| `confidence_threshold` | IntentClassifier | 0.70 | Min confidence for rule-based intent |
| `silence_thresh` | ShortRecorder (VAD) | 500 | RMS threshold for silence detection |
| `silence_duration` | ShortRecorder (VAD) | 3.0 seconds | How long silence triggers stop |
| `LISTEN_TIMEOUT` | src/main.py | 8.0 seconds | Max wait for user input after wake |
| `session_ttl` | SessionManager | 3600 seconds | Session expiration time |

### Environment Variables

```bash
# Required
DEEPGRAM_API_KEY="..."           # For ASR + TTS
Gemini_API_key="..."             # For LLM
REDIS_URL="rediss://default:TOKEN@HOST:PORT"  # For session storage (optional)
API_BASE_URL="https://..."       # Recipe backend API

# Optional
PORCUPINE_ACCESS_KEY="..."       # If not embedded in model
ENVIRONMENT="development|production"
```

---

## Error Handling & Fallbacks

### ASR Failure
```
Deepgram unavailable → WhisperASR (local Whisper model)
```

### Intent Classification Failure
- Rule confidence < threshold → LLM classification attempt
- LLM unavailable → Intent: UNKNOWN
- User input empty/silent → Resume previous audio

### Recipe Retrieval Failure
- No results found → Inform user, ask for different search
- API unavailable → Error message, attempt local fallback

### Response Generation Failure
- Structured JSON invalid → Fallback to plain text
- LLM timeout → Use cached response or generic message

### Session Storage Failure
- Redis unavailable → In-memory session storage (not persistent)
- On exit: session lost if Redis down

---

## Performance Characteristics

| Operation | Typical Time | Blocking |
|-----------|-------------|----------|
| Wake word detection | <50 ms per frame | No (background) |
| Recording + VAD | 3-5 sec (user depends) | Yes (main thread) |
| ASR (Deepgram) | 1-3 sec | Yes |
| Intent classification (rule-based) | 10-50 ms | Yes |
| Intent classification (LLM fallback) | 500-2000 ms | Yes |
| Recipe search (Milvus) | 100-500 ms | Yes |
| LLM response generation | 1-3 sec | Yes |
| TTS (Deepgram) | 1-5 sec (depends on text length) | Yes |
| Audio playback | Real-time | No (separate thread) |

**Total user experience**: ~8-15 seconds from wake word to first audio response (dominated by ASR + LLM).

---

## Conclusion

The BTP Voice Assistant uses a **modular pipeline architecture** with:
- **Separation of concerns**: Each component (ASR, intent, retrieval, etc.) is independent
- **Graceful fallbacks**: Multiple redundancy at each stage
- **Session awareness**: State persistence enables multi-turn conversations
- **Real-time responsiveness**: Background wake word detection + concurrent playback
- **Scalable design**: Easy to add new intents, retrieval methods, or LLM providers

This architecture balances **academic rigor** (formal components, clear data flow) with **practical usability** (error handling, performance tuning).
