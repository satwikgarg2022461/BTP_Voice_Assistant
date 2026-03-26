# Design Decisions & Rationale

This document explains the technical choices, trade-offs, and architectural decisions made during the development of the BTP Voice Assistant. It is structured to help academic evaluators understand the design philosophy and justification for each component selection.

---

## Overview

The BTP Voice Assistant balances **user experience** (low latency, natural interaction), **system reliability** (error handling, fallbacks), and **implementation feasibility** (available SDKs, cost constraints) across a speech-first conversational interface.

---

## 1. Wake Word Detection: Picovoice Porcupine

### Decision
Use **pre-trained Porcupine model** ("Hey Cook") for always-on wake word detection.

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|--------|
| **Porcupine (chosen)** | Low latency (<50ms), always-on ready, private (local), pre-trained | Proprietary (requires Picovoice subscription), fixed keyword | BEST for hands-free |
| Sphinx/PocketSphinx | Open-source, free | Outdated, poor accuracy, high false-positive rate | Poor UX |
| Google Snowboy | Cloud-connected, customizable | Discontinued (Google killed it), cloud-dependent | Not viable |
| WebRTC VAD + custom model | Complete control, open-source | Requires training data, complex tuning, high latency | Overkill for cooking env |
| Keyword spotting (small LLM) | Modern, flexible | GPU-heavy, 300+ms latency, high power consumption | Too slow for real-time |

### Rationale

1. **Always-on requirement**: Cooking is hands-busy; users cannot reach a button. Porcupine enables true push-to-talk without pushing.

2. **Fixed keyword acceptable**: "Hey Cook" is semantically fitting for a cooking assistant. Users won't need 10 different wake words.

3. **Local execution**: Processes audio on-device → privacy (no audio sent to cloud until ASR), low latency.

4. **Proven reliability**: Picovoice Porcupine is industry-standard (used by Alexa, Google Home prototypes).

5. **Low computational cost**: ~5% CPU on modern hardware, <1ms per frame.

### Trade-offs

- **Cost**: Picovoice subscription (~$0/month for hobby, $10-50/month for commercial)
- **Flexibility**: Cannot easily change wake word without retraining
- **Vendor lock-in**: Adds Picovoice as dependency

### Implementation Details

```python
WakeWordDetector(
    keyword_paths=["models/Hey-Cook_en_linux_v3_0_0.ppn"],
    sensitivities=[0.65],  # 0-1 scale; tuned for cooking noise
    device_index=-1  # auto-detect audio device
)
```

- **Sensitivity 0.65**: Balanced to detect natural speech while rejecting background chatter (pot sizzling, oven beeping)
- **Pre-roll buffer (1 sec)**: Captures start of user speech (wake word + first word)

---

## 2. Automatic Speech Recognition (ASR): Hybrid Approach

### Decision
**Primary**: Deepgram API (cloud-based)  
**Fallback**: OpenAI Whisper (local)

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|--------|
| **Deepgram API (primary)** | Fast (1-3 sec), high accuracy, supports cooking domain | Requires API key, $0.0043/min cost, internet dependent | BEST overall |
| **Whisper (fallback)** | Open-source, free, works offline, no vendor lock-in | Slow (~10 sec per 30 sec audio), high memory (2GB+ for large model) | GOOD fallback |
| Google Cloud Speech-to-Text | High accuracy, supports multiple languages | Expensive ($0.024/min), latency 2-5 sec | Too costly |
| Azure Speech Services | Good accuracy, enterprise support | Similar cost structure as Google | Similar trade-off |
| Sphinx/KALDI | Open-source, offline, free | Outdated (accuracy 60-70%), hard to deploy | Unacceptable quality |

### Rationale

1. **Hybrid approach**: Combines speed (Deepgram) with resilience (Whisper local)
   - Fast path: Deepgram cloud ASR (1-3 sec) for normal operation
   - Fallback: Whisper local (~10 sec) if Deepgram API unavailable or rate-limited

2. **Cooking domain considerations**:
   - Deepgram trained on diverse audio → handles kitchen noise (ventilation, sizzling)
   - Whisper also trained on noisy backgrounds from YouTube
   - Both better than older ASR systems for colloquial speech ("gimme", "how's")

3. **Cost-benefit**: 
   - Deepgram: ~$0.0043/min × 5 min/hour = ~$0.02/hour per user
   - Acceptable for prototype; scales to $3-5/month for 100 concurrent users

4. **Latency requirement**: <3 sec ASR latency maintains conversational flow
   - 3-5 sec total (record + ASR) feels responsive to user
   - >10 sec feels like system hung

### Trade-offs

- **API dependency**: Deepgram outage → fallback to slower Whisper
- **Network latency**: Adds 100-200ms for cloud roundtrip
- **Cost**: Variable depending on usage

### Implementation Details

```python
# Primary path (Deepgram)
asr = DeepgramASR(api_key=os.getenv("DEEPGRAM_API_KEY"))
transcription = asr.transcribe(audio_path)  # ~1-3 sec

# Fallback (Whisper local)
if not transcription or error:
    asr = WhisperASR(model_size="small")  # 577 MB model
    transcription = asr.transcribe(audio_path)  # ~10 sec
```

- **Whisper model size**: "small" (244M params) balances speed/accuracy
  - Larger models (base, medium) give +1-2% accuracy but +5-10 sec latency
  - Smaller models (tiny) at risk for poor quality

---

## 3. Intent Classification: Hybrid Rule-Based + LLM Fallback

### Decision
**Primary**: Rule-based patterns with regex  
**Fallback**: Google Gemini 2.5 Flash LLM

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|--------|
| **Rule-based (primary)** | Fast (10-50ms), deterministic, no API cost, explainable | Labor-intensive to maintain, errors on paraphrasing | Good for 80% of cases |
| **LLM only** | Flexible, handles paraphrasing, no rule maintenance | 500-2000ms latency, API costs, non-deterministic, hallucinations | Too slow as primary |
| **Hybrid (chosen)** | Fast path for common intents + LLM for edge cases | Maintenance burden (rules + LLM prompts) | BEST trade-off |
| Large BERT classifier | Fine-tuned on cooking domain, balanced latency | Requires training data, GPU for inference | Overkill complexity |
| Decision trees | Fast, interpretable | Limited generalization, manual tuning required | Restrictive |

### Rationale

1. **Cooking domain is well-defined**:
   - 17 intent types across search, navigation, QA, small-talk
   - Navigation intents (next, prev, repeat) have clear linguistic patterns
   - Rules can capture 80%+ of user inputs

2. **Latency critical**: Rules execute in <50ms vs. LLM's 500-2000ms
   - User perceives <100ms as instant; >300ms feels like lag
   - Rules used for common intents, LLM only on edge cases

3. **Cost optimization**:
   - Rules: $0/month
   - LLM: $0.075 per 1M input tokens (~5-10 tokens per query)
   - Fallback to LLM on only ~5-10% of inputs → saves 90% of LLM cost

4. **Explainability**:
   - Rules are transparent (pattern matching)
   - LLM fallback used for ambiguous/complex cases (acceptable non-determinism)

### Rule Examples

```python
Intent.NAV_NEXT: [
    {"pattern": r"\b(next|continue|forward|proceed|go ahead|move on)\b", "confidence": 0.95},
    {"pattern": r"\b(what'?s next|after that|then what)\b", "confidence": 0.90},
]

Intent.SEARCH_RECIPE: [
    {"pattern": r"\b(find|search|look for|show me|give me|i want|i need) (a |some |the )?(recipe|dish)\b", "confidence": 0.95},
]
```

### Trade-offs

- **Maintenance**: As system evolves, rules must be updated manually
- **Hard to extend**: New intents require careful pattern design
- **Misclassification**: Rules can false-positive (e.g., "next" in "Nexus" or "I'm next")

### Implementation Details

```python
classifier = IntentClassifier(confidence_threshold=0.7, use_llm_fallback=True)
intent, confidence, entities = classifier.classify(transcription)

# Confidence < 0.7: invoke LLM
if confidence < 0.7 and use_llm_fallback:
    intent, confidence = llm.classify_intent(transcription, context)
```

---

## 4. Recipe Retrieval: Milvus Vector Database + Semantic Search

### Decision
Use **Milvus vector database** with **SentenceTransformer embeddings** for semantic search.

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|--------|
| **Milvus + embeddings (chosen)** | Fast (100-500ms), semantic understanding, scales to millions | Setup complexity, embedding model memory (2GB) | BEST for quality |
| Traditional full-text search (ES/Solr) | Fast, proven, low memory | Lacks semantic understanding, poor for synonyms | Poor UX (synonyms miss) |
| SQL database with LIKE queries | Simple, low overhead | Very poor relevance, cannot rank by meaning | Unacceptable |
| LLM-only retrieval | Contextual, flexible | Slow (1-3 sec per query), expensive | Too slow/costly |
| Hybrid (BM25 + embeddings) | Best of both worlds | More complex, dual ranking | Over-engineered for this scale |

### Rationale

1. **Semantic understanding**:
   - User query: "How do I make pasta?"
   - Traditional search: Exact match "pasta" → ok
   - Semantic search: Understands "pasta" ≈ "spaghetti" ≈ "noodles" → better
   - Embedding-based: Maps to 384-D space, finds closest neighbors

2. **Scalability**: Milvus efficiently indexes millions of recipes
   - In-memory index: typical cooking dataset ~10K recipes (10K × 384 floats = ~15 MB) = fast
   - Disk-based fallback for larger datasets

3. **No training required**: SentenceTransformer pre-trained on general text
   - No need to fine-tune on cooking domain
   - Transfers well (cooking language overlaps with general English)

4. **Deterministic results**: Vector search gives consistent top-K results (unlike LLM)

### Embedding Model Selection

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

- **all-MiniLM-L6-v2**:
  - 384-D output (good compression-quality trade-off)
  - 22M parameters (fits on CPU, 87 MB)
  - Fast inference (~10-50ms per query)
  - Pre-trained on SNLI, similarity datasets

- Alternatives:
  - `all-mpnet-base-v2`: Higher quality but slower (1000+ MB, 109M params)
  - `all-distilroberta-v1`: Faster but lower quality (333-D)
  - `gte-small`: Cooking-specific? Not available pre-trained

### Trade-offs

- **Semantic drift**: LLM-like understanding has blind spots (e.g., "healthy" might not match "low-fat" perfectly)
- **Setup complexity**: Milvus requires separate service
- **Memory**: 384-D embeddings for 100K recipes = ~150 MB RAM

### Implementation Details

```python
retriever = RecipeRetriever(
    db_path="recipes_demo.db",
    collection_name="recipes_collection",
    model_name='all-MiniLM-L6-v2'
)

results = retriever.search_recipes("pasta", top_k=5)
# Returns: [{recipe_id: int, similarity_score: 0-1}, ...]

full_recipe = retriever.fetch_full_recipe_details(results[0]['recipe_id'])
```

---

## 5. LLM Choice: Google Gemini 2.5 Flash

### Decision
Use **Google Gemini 2.5 Flash** for structured JSON response generation and fallback intent classification.

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|--------|
| **Gemini 2.5 Flash (chosen)** | Fast (1-3 sec), good quality, JSON output support, free tier available | Google API dependency, cost per token | BEST balance |
| GPT-4o (OpenAI) | Excellent quality, multimodal | Expensive ($0.15/1K input), slower (3-5 sec) | Too costly |
| GPT-4o mini | Cheaper than GPT-4o, fast | Still somewhat expensive | Acceptable alternative |
| Llama 2 (Meta, local) | Free, open-source, no API cost | Slow on CPU (5-10 sec), quality lower than GPT | Too slow |
| Mixtral 8x7B | Efficient mixture-of-experts, good quality | Still requires GPU, ~40B tokens/sec = 2-3 sec | Marginal benefit over cloud |
| Claude 3 Haiku (Anthropic) | Fast, good quality | API cost similar to GPT, less availability | Similar to Gemini |

### Rationale

1. **Latency**: 1-3 sec for LLM response is acceptable for fallback + conversational responses
   - Not in critical path for recipe navigation (off-loaded to structured JSON)
   - Users expect slight delay for "thinking" (LLM inference)

2. **JSON structured output**:
   - Gemini has native JSON mode (ensures valid JSON)
   - Compared to text output → requires parsing/validation

3. **Cost-benefit**:
   - Tier 1: $0/month (free),  ~50 requests/min
   - Tier 2: $0.075 per 1M input tokens (~400 recipes @ 100 tokens each = $0.003/batch)
   - For prototype/research: essentially free

4. **Quality for cooking domain**:
   - Gemini trained on diverse data including recipes
   - Can handle ingredient substitutions, cooking questions, small talk

5. **Fallback availability**:
   - Intent classification fallback: every query needs response
   - If Gemini unavailable → fallback to UNKNOWN intent (graceful degradation)

### Model Selection

```python
llm = RecipeLLM(model_name="gemini-2.5-flash")
# Alternatives if needed:
# - "gemini-1.5-flash": Older, slightly slower
# - "gemini-2.0-flash": Newer, might be faster
```

- **gemini-2.5-flash**:
  - ~1-2 sec latency
  - Good balance of speed vs. quality
  - JSON output support
  - Better than "flash-exp" (experimental)

### Trade-offs

- **Hallucination risk**: LLM can generate plausible-sounding but incorrect cooking info
  - Mitigated: Use only for intent classification (deterministic mapping) and small-talk
  - Recipe responses use structured templates + API data (not LLM generation)

- **API dependency**: Google outage → no LLM fallback for intent classification
  - Fallback to UNKNOWN intent (system still works, just less smart)

### Implementation Details

```python
response = llm.generate_recipe_response(
    user_query="How do I make pasta?",
    recipe_results=[...],
    return_json=True
)

# Returns JSON with "greeting", "ingredients", "steps", "closing"
# - Can fallback to plain text if JSON parsing fails
# - Extracted manually if LLM returns markdown code blocks
```

---

## 6. Session Storage: Upstash Redis vs. In-Memory

### Decision
**Primary**: Upstash Redis (cloud-hosted)  
**Fallback**: In-memory dictionary

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|--------|
| **Upstash Redis (primary)** | Persistent, multi-device support, simple API, free tier | Requires internet, API latency (~50-100ms), dependency | BEST for persistence |
| **In-memory (fallback)** | Zero latency, no external dependency, simple | Lost on restart, single-device only | ACCEPTABLE fallback |
| Traditional DB (PostgreSQL) | ACID, reliable, widely used | Overkill for simple KV, setup complexity | Over-engineered |
| DynamoDB (AWS) | Scalable, serverless | Expensive, AWS vendor lock-in, complexity | Too much for prototype |
| Local SQLite | Simple, no external dependency | Not distributed, limited session sharing | Single-user only |

### Rationale

1. **Session persistence**: Users can pause recipe, close app, reopen → continue where left off
   - Redis provides this with TTL (1 hour default)
   - Optional (fallback to in-memory if unavailable)

2. **Multi-device support**: Redis enables household scenarios
   - One session shared across devices (future expansion)
   - Not essential for current prototype

3. **Simplicity**: Redis is KV store, perfect for session data
   - Vs. traditional DB: no schema definition, migrations
   - vs. files: simpler than managing JSON files

4. **Cost**: Upstash free tier:
   - 10,000 commands/day
   - ~100 commands per user-session (state updates, history)
   - Supports ~100 daily users free

### Trade-offs

- **API latency**: +50-100ms for Redis calls (vs. 1-2ms local)
  - Acceptable since not in critical path
  - Session updates happen after user response spoken

- **Internet dependency**: No Redis → in-memory fallback (works but non-persistent)

### Implementation Details

```python
session_mgr = SessionManager(session_ttl=3600)

# Primary: Upstash Redis (if REDIS_URL configured)
if os.getenv("REDIS_URL"):
    session = session_mgr.create_session(...)  # Persisted
else:
    session = session_mgr.create_session(...)  # In-memory fallback
```

---

## 7. Text-to-Speech: Deepgram TTS

### Decision
Use **Deepgram TTS API** for speech synthesis.

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|--------|
| **Deepgram TTS (chosen)** | High quality voices, fast (1-5 sec), consistent with ASR | Cost ($0.003/1K chars), cloud dependency | BEST quality |
| Google Cloud TTS | High quality, multiple languages | Similar cost, latency 2-3 sec | Similar quality, slightly slower |
| Azure Speech Services | Excellent quality, on-prem option | Similar cost, setup complexity | Similar to Google |
| Festival/eSpeak (local) | Free, open-source, no internet | Poor voice quality, robotic sound | Unacceptable user experience |
| Pico TTS | Fast, lightweight | Very poor quality | Unacceptable |
| Glow-TTS (local, fast) | Open-source, good quality, fast | Requires setup/GPU for speed | Viable alternative |

### Rationale

1. **User experience**: Natural-sounding voice essential for cooking assistant
   - Robotic voice (Festival) → users think system is broken
   - Human-like voice → more engaging, easier to follow instructions

2. **Consistency**: Same vendor (Deepgram) for both ASR and TTS
   - Know quality level, single API integration
   - Pricing consistent

3. **Speed**: 1-5 sec for ~500 characters = acceptable latency
   - Typical recipe greeting: 100-200 chars = 1 sec
   - Typical step: 50-150 chars = 0.5-1 sec

4. **Cost**: $0.003 per 1K characters
   - ~100K chars per recipe (greeting + 8 steps + closing)
   - 1000 recipes processed = $0.30
   - Acceptable for prototype

### Voice Model Selection

```python
tts = RecipeTTS(model="aura-asteria-en")
```

- **aura-asteria-en**: Natural, friendly tone suitable for cooking instructions
- Alternatives: `aura-luna-en` (calm), `aura-orpheus-en` (warm)

### Trade-offs

- **Cost**: ~$0.003/1K chars vs. free local TTS
  - For 1000 users, 1 recipe each: ~$0.30/day
  - Netflix ~$15/month; this is negligible

- **Internet dependency**: No cloud → fallback to local TTS (poor quality)
  - Acceptable compromise: "At least something plays"

### Implementation Details

```python
# Chunk long text into sentences for optimal TTS
chunks = tts.split_into_chunks(
    response_text,
    max_chunk_length=300  # ~1-2 sentences
)

# Generate MP3 for each chunk
mp3_paths = [tts.generate_speech(chunk) for chunk in chunks]

# Stream playback
player.play_chunks(mp3_paths)
```

---

## 8. Audio Playback: ChunkedAudioPlayer

### Decision
Implement **concurrent playback + recording thread** to enable wake word interruption during TTS.

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|--------|
| **ChunkedAudioPlayer (chosen)** | True concurrency, wake word detection during playback, responsive | Thread management complexity, synchronization bugs | BEST UX |
| Sequential playback | Simple, predictable | Cannot interrupt; wait entire response before new command | Poor UX |
| Single-threaded with polling | Simple | Blocking audio playback, miss wake words during TTS | Unacceptable |
| Multi-process | Better isolation | Over-kill, IPC complexity | Unnecessary |

### Rationale

1. **Cooking scenario**: User may interrupt ("Hold on, wait!" or "next step!")
   - Sequential playback forces waiting entire response
   - Concurrent: wake word detected immediately, pause current speech

2. **User control**: Responsiveness critical for hands-busy environment
   - If user says "next", expect immediate pause/new response
   - Not <500ms latency → feels sluggish

3. **Architecture**: Background wake word thread already exists → reuse threading model

### Trade-offs

- **Complexity**: Thread synchronization, race conditions
  - Mitigated: Use thread-safe queues, event signaling

- **Resource usage**: Extra thread (~1-2 MB stack)
  - Negligible on modern hardware

### Implementation Details

```python
player = ChunkedAudioPlayer()

# Main thread:
listener_thread = threading.Thread(target=wake_word_detection)
listener_thread.daemon = True
listener_thread.start()

# Later...
player.play_chunks(mp3_chunks)  # Non-blocking, background playback

# If user dares: recorded audio triggers pause
wake_word_detected()
player.pause()
# Process new command...
player.resume()  # or start new response
```

---

## 9. Database Format: CSV vs. SQL

### Decision
Use **CSV files** for recipe data (chunks, ingredients, metadata).

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|--------|
| **CSV (chosen)** | Simple, human-readable, no schema migration, version control friendly | No querying (read entire file), poor for updates | BEST for read-heavy static data |
| Relational DB (SQLite/PostgreSQL) | ACID, querying, indexing, updates | Overkill for static data, schema management | Over-engineered |
| NoSQL (MongoDB) | Flexible schema, JSON native | Overkill, server dependency | Over-engineered |
| Milvus only | Vector search native | Cannot store raw recipe text (too large for embedding) | Missing features |

### Rationale

1. **Read-heavy workload**: Recipes rarely updated (static dataset)
   - CSV sufficient; no need for transactional support

2. **Data size**: ~50 MB total CSVs fit in memory or disk cache
   - Load once on startup → fast lookups

3. **Version control**: CSV files in Git → reproductive research
   - Vs. DB blobs that cannot be diffed

4. **Simplicity**: No database setup/migration
   - Lower friction for new developers

### Trade-offs

- **Scalability**: Millions of recipes → CSV impractical (memory issues)
  - Solution: Migrate to Milvus + API for future scale

- **Concurrent updates**: Not supported by CSV format
  - Acceptable: recipes updated offline, deployed as new CSV

### Implementation Details

```python
# Load at startup
self.chunks = pd.read_csv("data/chunks.csv")
self.recipes = pd.read_csv("data/recipes.csv")
self.food_dict = pd.read_csv("data/food_dictionary.csv")

# In-memory lookups
chunk = self.chunks[self.chunks['recipe_id'] == 4521]
```

---

## 10. Threading Model: Daemon Thread for Wake Word

### Decision
Implement **always-on daemon thread** for wake word detection.

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|--------|
| **Daemon thread (chosen)** | Always listening, non-blocking main thread, simple coordination | Thread complexity, no graceful shutdown | BEST for responsiveness |
| Event-driven (asyncio) | Elegant, scalable | Steep learning curve, require async throughout | Over-complex for this app |
| Polling (main thread) | Single-threaded simplicity | Blocking, cannot listen while TTS | Poor UX |
| Process-based (multiprocessing) | Isolation, independent runtime | IPC complexity, resource overhead | Overkill |

### Rationale

1. **"Always on"**: Porcupine must continuously process incoming audio
   - Cannot pause during TTS playback (miss wake words)
   - Daemon thread ideal: background task, non-blocking

2. **Simplicity**: Python threading is straightforward for I/O-bound tasks
   - Porcupine listening = I/O-bound (waiting for mic input)
   - Async would be over-engineered

3. **Responsiveness**: User expects <100ms reaction to "Hey Cook" even during playback
   - Daemon thread delivers this

### Trade-offs

- **No graceful shutdown**: Daemon threads don't block program exit
  - Acceptable: App is CLI (not a service), exit on Ctrl+C is ok

- **Synchronization complexity**: Shared variables (detected_event) require thread safety
  - Mitigated: Use thread-safe Event objects

### Implementation Details

```python
class WakeWordDetector:
    def start(self):
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
    
    def _detection_loop(self):
        while True:
            frame = self.recorder.read()
            result = self.porcupine.process(frame)
            if result:
                self.detected_event.set()
                sleep(0.1)  # Prevent rapid re-triggers
```

---

## 11. Configuration Management: Environment Variables

### Decision
Use **environment variables** (.env file) for configuration.

### Rationale

1. **Secrets management**: API keys never hardcoded
   - `.env` in `.gitignore` → prevents accidental commit

2. **Environment-specific tuning**:
   - Development: `VAD_SILENCE_THRESHOLD=500`
   - Production: `VAD_SILENCE_THRESHOLD=600` (noisier environment)
   - No code changes needed

3. **Standard practice**: .env paradigm widely understood

### Trade-offs

- **Not validated at startup**: Typos in `.env` discovered at runtime
  - Solution: Explicit env var checks with helpful error messages

---

## 12. Error Handling Strategy: Graceful Degradation

### Decision
Implement **cascading fallbacks**: try primary → fallback → degrade gracefully

### Examples

1. **ASR Failure**
   ```
   Try: Deepgram API
   Fallback: Whisper local
   Degrade: "Sorry, couldn't hear you. Please repeat."
   ```

2. **Recipe Not Found**
   ```
   Result: Milvus finds 0 recipes
   Message: "I couldn't find recipes matching that. Try 'pasta' or 'omelet'."
   ```

3. **Session Storage Down**
   ```
   Try: Upstash Redis
   Fallback: In-memory dict
   Degrade: No persistence (session lost on exit, acceptable)
   ```

### Rationale

- **Robustness**: Single point of failure doesn't crash system
- **User experience**: Clear error messages instead of cryptic stack traces

---

## Conclusion: Design Philosophy

The BTP Voice Assistant prioritizes:

1. **User experience** over perfection
   - Latency critical; accuracy good-enough
   - Fallbacks enable continued operation

2. **Simplicity** over flexibility
   - Fixed wake word OK
   - CSV data OK for ~10K recipes
   - Rule-based intent classification covers 80%

3. **Cost efficiency** over scale
   - Cloud APIs for quality (Deepgram, Gemini)
   - Free fallbacks where possible (Whisper, in-memory session)
   - Acceptable for prototype / small user base

4. **Reproducibility** for research
   - CSV-based data in version control
   - Deterministic rule-based classification
   - Documented decisions (this file)

This is a **research system** balancing academic rigor with practical usability, not production infrastructure.
