# Core Algorithms & Detailed Explanations

This document provides in-depth explanations of the key algorithms and computational techniques used in the BTP Voice Assistant.

---

## 1. Voice Activity Detection (VAD) Algorithm

### Overview

The **ShortRecorder VAD** detects when the user stops speaking and automatically ends recording. It uses **RMS (Root Mean Square) energy** of audio frames to distinguish speech from silence.

### Algorithm

```
Input: Continuous audio stream (16-bit PCM at 16 kHz)
Output: .wav file containing user utterance

1. Initialize:
   - pre_roll_buffer: circular buffer of 16K samples (1 sec @ 16 kHz)
   - silence_threshold: RMS energy threshold (e.g., 500)
   - silence_counter: frames of consecutive silence
   - max_silence_duration: frames before stopping (e.g., 3 sec = ~93 frames)

2. Pre-roll phase (fill silence detection buffer):
   for i in 1 to pre_roll_frames:
       frame = recorder.read()  // 512 samples
       pre_roll_buffer.append(frame)

3. Recording phase:
   silence_counter = 0
   audio_frames = []
   
   while True:
       frame = recorder.read()  // 512 samples
       audio_frames.append(frame)
       
       // Compute RMS energy
       rms_value = sqrt(mean(frame^2))
       
       // Update silence counter
       if rms_value < silence_threshold:
           silence_counter += 1
       else:
           silence_counter = 0  // reset on any energy
       
       // Check termination
       if silence_counter >= max_silence_frames:
           break  // User done speaking

4. Finalize:
   full_audio = pre_roll_buffer + audio_frames
   save_wav(full_audio, output_path)
   return output_path
```

### Parameters & Tuning

| Parameter | Default | Unit | Purpose |
|-----------|---------|------|---------|
| `sample_rate` | 16000 | Hz | Must match ASR expectation |
| `frame_length` | 512 | samples | ~32 ms @ 16 kHz; matches Porcupine |
| `pre_roll_secs` | 1.0 | sec | Capture audio before trigger word |
| `silence_thresh` | 500 | RMS units | Energy threshold for "silence" |
| `silence_duration` | 3.0 | sec | Duration of silence before stop |

### Tuning for Cooking Environment

**Problem**: Kitchen noise (ventilation fan, sizzling pan) can trigger false silence.

**Solution 1**: Increase `silence_thresh` (e.g., 500 → 700)
```
+ Pro: Less sensitive to ambient noise
- Con: May cut off soft-spoken instructions
```

**Solution 2**: Increase `silence_duration` (e.g., 3.0 → 4.5 sec)
```
+ Pro: Waits longer for user to resume
- Con: Slower response (user has to wait)
```

**Typical cooking setting**:
```python
recorder = ShortRecorder(
    silence_thresh=600,      # Increased for kitchen noise
    silence_duration=3.5     # Balance between responsiveness and robustness
)
```

### Example Execution

```
Frame 0-30:    Pre-roll fill (1 sec)
Frame 31:      User says "Hey Cook" (wake word, already detected)
Frame 32-35:   User says "next step" (HIGH energy: RMS=1200)
               silence_counter = 0 (reset)
Frame 36-40:   Kitchen fan noise (MEDIUM energy: RMS=550)
               silence_counter += 1 (but RMS > 500, so reset to 0)
Frame 41-100:  Silence (LOW energy: RMS=300)
               silence_counter = 1, 2, 3, ...
Frame 104:     silence_counter = 93 >= max_silence (93 frames = 3 sec)
               STOP RECORDING
```

### Advantages & Limitations

**Advantages:**
- Simple, real-time computation (no buffering delay)
- Works well for discrete utterances (recipe commands)
- Resistant to background noise if threshold tuned properly

**Limitations:**
- Cannot distinguish speech from other loud noises (alarm beeping)
- Requires manual tuning for each environment
- No acoustic model (vs. sophisticated VAD like WebRTC)

---

## 2. Intent Classification Algorithm

### Overview

A two-stage pipeline:
1. **Rule-based matching** (fast, deterministic)
2. **LLM fallback** (accurate, slower, only on low confidence)

### Stage 1: Rule-Based Classification

```
Input: user_input (string)
Output: (Intent, confidence, entities)

Algorithm:
1. For each intent_type in [NAV_NEXT, NAV_PREV, ..., UNKNOWN]:
   best_match = None
   best_confidence = 0
   
   for pattern_dict in intent_patterns[intent_type]:
       pattern_regex = pattern_dict["pattern"]
       base_confidence = pattern_dict["confidence"]
       
       if regex_match(pattern_regex, user_input):
           // Found matching pattern
           confidence = base_confidence
           
           // Apply context boosts
           if current_intent == intent_type:  // continued same intent
               confidence += CONTEXT_BOOST_SMALL
           
           if confidence > best_confidence:
               best_confidence = confidence
               best_match = intent_type

2. Extract entities (if any):
   if intent_type == NAV_GO_TO:
       entities["target_step"] = extract_number(user_input)
   elif intent_type == QUESTION:
       entities["ingredient"] = extract_ingredient(user_input)

3. Return (best_match, best_confidence, entities)
```

### Pattern Examples

```python
Intent.NAV_NEXT:
  [
    {"pattern": r"\b(next|continue|forward|proceed)\b", "confidence": 0.95},
    {"pattern": r"\b(what'?s next|then what)\b", "confidence": 0.90},
    {"pattern": r"\b(skip|move forward)\b", "confidence": 0.85},
  ]

Intent.SEARCH_RECIPE:
  [
    {"pattern": r"(find|search|look for|show me).*(recipe|dish)", "confidence": 0.95},
    {"pattern": r"how (do|to) (make|cook|prepare)", "confidence": 0.90},
    {"pattern": r"cook|prepare", "confidence": 0.60},  # Very general
  ]
```

### Pattern Matching Examples

```
User input: "next"
→ Match rule NAV_NEXT: "pattern: \b(next|continue|forward)\b"
→ confidence = 0.95
→ Return (Intent.NAV_NEXT, 0.95, {})

User input: "what should i do"
→ No rule matches
→ best_confidence = 0 (no match)
→ confidence >= threshold? NO (0 < 0.7)
→ Proceed to Stage 2 (LLM fallback)

User input: "make pasta"
→ Match rule SEARCH_RECIPE: "pattern: cook|prepare"
→ But rule has 0.60 confidence (very general)
→ Also match "make" (weak pattern)
→ best_confidence = 0.60
→ 0.60 < threshold (0.70)? Combined with context boost (0.05) → 0.65 < 0.70
→ Proceed to Stage 2
```

### Stage 2: LLM Fallback

Triggered when: `rule_confidence < confidence_threshold`

```
Input: user_input, context (optional)
Output: (Intent, confidence, entities) — from LLM

Algorithm:
1. Prompt construction:
   system_prompt = """
   You are a cooking assistant intent classifier.
   Classify the user input into one of these intents:
   - NAV_NEXT: request to continue recipe
   - NAV_PREV: go back to previous step
   - SEARCH_RECIPE: search for a recipe
   - QUESTION: ask a cooking question
   - SMALL_TALK: greeting or casual conversation
   - ... (full list of 17 intents)
   
   Respond in JSON:
   {"intent": "NAV_NEXT", "confidence": 0.85, "entities": {...}}
   """
   
   user_prompt = f"User said: '{user_input}'. Classify this intent."

2. Call Gemini API:
   response = gemini.generate_content(
       system_prompt + user_prompt
   )  // ~500-2000ms latency

3. Parse JSON response:
   result = json.parse(response.text)
   confidence = result["confidence"]  # LLM-estimated confidence
   entities = result.get("entities", {})
   intent = Intent[result["intent"]]

4. Return (intent, confidence, entities)
```

### Examples

```
Stage 1 returns: confidence = 0.50 < threshold (0.70)

Stage 2 LLM prompt: "User said: 'what should i do?'"
LLM response: {
    "intent": "CLARIFY",
    "confidence": 0.70,
    "entities": {},
    "reasoning": "User asking for clarification/guidance"
}

Final return: (Intent.CLARIFY, 0.70, {})
```

### Confidence Score Interpretation

| Confidence | Interpretation |
|-----------|-----------------|
| 0.90-1.0 | Very confident (e.g., exact phrase match "next") |
| 0.75-0.90 | Confident (e.g., keyword match "what's next") |
| 0.70-0.75 | Borderline (may need context to disambiguate) |
| <0.70 | Low (ambiguous, LLM needed) |

### Optimization: Context Boost

If user previously used intent X, and now matches a rule for X:
```
confidence += CONTEXT_BOOST_SMALL (0.05)
```

**Example:**
```
Turn 1: User says "next" → Intent.NAV_NEXT (confidence 0.95)
Turn 2: User says "go ahead"
  - Rule matches: NAV_NEXT with confidence 0.85
  - Context boost applied: 0.85 + 0.05 = 0.90
  - Return (Intent.NAV_NEXT, 0.90, {})
```

This captures "continued intent" patterns (user repeating same intent).

### Complexity Analysis

**Stage 1 (rule-based):**
- Time: O(P × N) where P = number of patterns (~50), N = input length (~30 words)
- Typical: ~10-50 ms (very fast)

**Stage 2 (LLM):**
- Time: O(1) but 500-2000 ms due to API latency
- Called only on ~5-10% of inputs → ~90% of calls remain fast

---

## 3. Semantic Search & Embedding Algorithm

### Overview

Recipes are embedded into a 384-dimensional vector space. User queries are embedded similarly, and nearest neighbors in this space are returned.

### Algorithm

```
Offline (one-time):
1. For each recipe in database:
   text = recipe_title + recipe_description + ingredients + instructions
   embedding = SentenceTransformer.encode(text)  // 384-D vector
   milvus.insert(recipe_id, embedding)

Online (per query):
2. user_query = "how do i make pasta?"
   query_embedding = SentenceTransformer.encode(user_query)  // 384-D
   
3. Milvus search:
   top_k_results = milvus.search(
       query_embedding,
       top_k=5,               // return 5 nearest neighbors
       metric="cosine"        // cosine similarity
   )
   
   // Results sorted by similarity (highest first)
   // similarity ∈ [0, 1] where 1.0 = identical

4. Return: [
     {recipe_id: 4521, similarity: 0.892},
     {recipe_id: 2134, similarity: 0.856},
     ...
   ]
```

### Embedding Model: SentenceTransformer all-MiniLM-L6-v2

**Architecture:**
- Input: Any text (up to ~512 tokens)
- Output: 384-D vector (normalized, -1 to +1)
- Model size: ~87 MB (fits any device)
- Speed: ~20-50 ms per text

**Example embeddings:**
```python
texts = [
    "how do i make pasta?",
    "spaghetti recipe",
    "noodles with garlic",
    "how do i bake a cake?"
]

embeddings = [
    [0.12, -0.45, 0.03, ...],  # 384-D
    [0.11, -0.43, 0.05, ...],  # Very similar to 1st
    [0.13, -0.44, 0.02, ...],  # Close to 1st and 2nd
    [0.02,  0.38, -0.20, ...], # Dissimilar (cake not pasta)
]

# Cosine similarity between embeddings[0] and embeddings[1]:
similarity = dot_product(embeddings[0], embeddings[1]) / (norm(0) * norm(1))
          ≈ 0.892  (very high, indicating semantic match)
```

### Similarity Metric: Cosine Distance

**Formula:**
```
similarity = (A · B) / (||A|| × ||B||)

where:
  A · B     = sum of element-wise products
  ||A||     = length (L2 norm) of vector A
  similarity ∈ [0, 1]
    0 = orthogonal (completely different)
    1 = identical
```

**Why cosine?**
- Invariant to magnitude (doesn't care about text length)
- Natural for high-dimensional spaces (more robust than Euclidean)
- Fast to compute (dot product)

### Practical Example

```
Query: "how do i make pasta?"
Query embedding: q = [0.12, -0.45, 0.03, ...]

Recipe 1: "Spaghetti Aglio e Olio"
Recipe embedding: r1 = [0.11, -0.43, 0.05, ...]
similarity(q, r1) = 0.892  ← High match!

Recipe 2: "Chocolate Cake"
Recipe embedding: r2 = [0.02, 0.38, -0.20, ...]
similarity(q, r2) = 0.12   ← Low match (different domain)

Top-5 results:
  1. Spaghetti Aglio e Olio (0.892)
  2. Fettuccine Alfredo (0.856)
  3. Pasta Carbonara (0.843)
  4. Penne Bolognese (0.821)
  5. Ravioli Filling (0.798)
```

### Why SentenceTransformer Works for Cooking

The model is pre-trained on general semantic similarity, but transfers well to cooking because:

1. **Ingredient synonyms**: 
   - "pasta" ≈ "spaghetti" ≈ "noodles" (all have similar embeddings)

2. **Recipe paraphrasing**:
   - "How do I make pasta?" ≈ "Recipe for spaghetti"
   - (Both map to similar region in embedding space)

3. **Common sense**:
   - "olive oil" closer to "cooking oil" than "motor oil"
   - (Semantic relationships preserved from training data)

### Limitations

1. **Out-of-domain terms**:
   - Rare ingredients or techniques may not have representative embeddings
   - Fallback: Exact keyword match in recipe database

2. **Semantic drift**:
   - "healthy" might not strongly match "low-fat"
   - (Requires post-ranking with recipe metadata: calories, nutrients)

3. **Scale**:
   - Works well for ~10K recipes
   - 1M+ recipes may require approximate nearest neighbor algorithms (e.g., HNSW)

---

## 4. Recipe Navigation Algorithm

### Overview

Recipes are divided into **chunks** (contiguous steps), and navigation maintains:
- Current chunk index
- Current step index (global, across chunks)
- ingredients section (step 0)

### Data Structure

```
Recipe: Spaghetti Aglio e Olio (recipe_id: 4521, total_steps: 8)

Chunk 1: steps 1-3, "Bring water to boil. Heat oil. Add garlic."
Chunk 2: steps 4-6, "Cook pasta. Reserve water. Combine pasta and oil."
Chunk 3: steps 7-8, "Toss together. Serve."

Navigation state:
  step_index: 1-8 (global step number)
  chunk_index: 1-3 (which chunk)
  section: "ingredients" (0) | "steps" (1-8) | "done" (>8)
```

### Navigation Commands

#### 1. NAV_NEXT: Go to next step

```
Algorithm:
if current_step < total_steps:
    current_step += 1
    current_chunk = find_chunk_for_step(current_step)
    return NavigationResult(
        success=True,
        step_index=current_step,
        chunk_index=current_chunk,
        text=fetch_step_text(current_step),
        section="steps",
        is_last_step=(current_step == total_steps)
    )
else:
    return NavigationResult(success=False, message="Recipe complete")

Time complexity: O(log C) where C = number of chunks (binary search)
```

#### 2. NAV_GO_TO: Jump to specific step

```
Algorithm:
target_step = extract_step_number(user_input)  // e.g., "go to step 5"

if 1 <= target_step <= total_steps:
    current_step = target_step
    current_chunk = find_chunk_for_step(target_step)
    return NavigationResult(
        success=True,
        step_index=target_step,
        chunk_index=current_chunk,
        text=fetch_step_text(target_step),
        section="steps",
        message=f"Jumped to step {target_step}"
    )
else:
    return NavigationResult(success=False, message="Step out of range")
```

#### 3. NAV_REPEAT_INGREDIENTS: Show all ingredients

```
Algorithm:
ingredients = fetch_all_ingredients(recipe_id)
ingredient_text = "You'll need: " + ", ".join(ingredients)

return NavigationResult(
    success=True,
    step_index=0,  // special value for ingredients
    chunk_index=0,
    text=ingredient_text,
    section="ingredients",
    extra={"ingredient_count": len(ingredients)}
)
```

#### 4. NAV_REPEAT: Repeat current step

```
Algorithm:
current_step_text = fetch_step_text(current_step_index)

return NavigationResult(
    success=True,
    step_index=current_step_index,
    chunk_index=current_chunk_index,
    text=current_step_text,
    section="steps",
    message="Repeating current step"
)

Time complexity: O(1) (cached current step)
```

### Session State Update

After each navigation, session is updated:

```python
session['current_step'] = navigation_result.step_index
session['current_chunk'] = navigation_result.chunk_index
session['last_activity'] = datetime.now()

# Mark step as "spoken"
session['steps_spoken'][navigation_result.step_index - 1] = True
```

### Example Sequence

```
Recipe started (session created):
  current_step = 0, current_chunk = 0, section = "ingredients"

User: "next"
  → current_step = 1, current_chunk = 1
  → Output: "Start by bringing a large pot of salted water..."

User: "next"
  → current_step = 2, current_chunk = 1
  → Output: "While the water is heating..."

User: "go to step 5"
  → current_step = 5, current_chunk = 2
  → Output: "Reserve 1 cup of pasta water, then drain."

User: "repeat"
  → current_step = 5, current_chunk = 2 (unchanged)
  → Output: (same as previous, step 5 again)

User: "ingredients"
  → current_step = 0, current_chunk = 0, section = "ingredients"
  → Output: "You'll need: 1 pound spaghetti, 6 cloves garlic, 0.5 cup olive oil..."

User: "next"
  → current_step = 1, current_chunk = 1, section = "steps"
  → Output: "Start by bringing a large pot of salted water..."
```

---

## 5. Chunk Boundary Detection

### Problem

Recipes must be split into navigable chunks, but chunk boundaries must respect:
- Natural breaks (ingredient list → cooking steps)
- Reasonable step count per chunk (3-6 steps)
- Timing (don't chunk in middle of multi-step instruction)

### Algorithm

```
Input: Recipe with N instructions
Output: C chunks with step ranges

0. Parse instructions into sentences
   (Deepcopy to avoid splitting mid-sentence)

1. Heuristic partitioning:
   chunk_size = 3-5 steps per chunk  (configurable)
   
   for i in 0 to N by chunk_size:
       start_step = i
       end_step = min(i + chunk_size, N)
       chunk = {
           start_step: start_step,
           end_step: end_step,
           text: concat(steps[start_step:end_step])
       }
       chunks.append(chunk)

2. Boundary refinement (optional):
   - If chunk ends mid-instruction (contains "then" or "and then"):
       extend to next sentence
```

### Example

```
Recipe: Pasta (8 steps)

Raw partitioning (3 steps per chunk):
  Chunk 1: steps 1-3 ✓ OK
  Chunk 2: steps 4-6 ✓ OK
  Chunk 3: steps 7-8 ✓ OK

Result: 3 balanced chunks
```

---

## Conclusion

These algorithms balance:
- **Performance**: Fast inference (rules <50ms)
- **Accuracy**: Fallbacks (LLM when rules uncertain)
- **Robustness**: Multiple error recovery paths
- **Simplicity**: Interpretable decisions (regex rules, vector search)

Together, they enable a responsive voice interface for hands-busy cooking environments.
