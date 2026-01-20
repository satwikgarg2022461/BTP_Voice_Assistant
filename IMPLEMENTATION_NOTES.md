# Structured JSON LLM Implementation Summary

## Overview
Successfully updated `modules/llm.py` to return structured JSON responses instead of plain text, enabling better tracking of ingredients and steps spoken during recipe narration.

## Changes Made

### 1. **Updated Imports**
- Added `json`, `re` for JSON parsing and regex operations
- Added type hints: `Dict`, `List`, `Optional`, `Union` from typing

### 2. **Modified `generate_recipe_response()` Method**

**New Signature:**
```python
def generate_recipe_response(self, user_query: str, recipe_results: List[Dict], 
                             return_json: bool = True) -> Union[Dict, str]
```

**Returns:**
```json
{
  "greeting": "Welcome message with recipe name",
  "ingredients": [
    {"text": "ingredient description", "spoken": false},
    ...
  ],
  "steps": [
    {"step_num": 1, "text": "step description", "spoken": false},
    ...
  ],
  "closing": "Encouraging closing message"
}
```

**Key Features:**
- Returns structured JSON by default (`return_json=True`)
- Can fallback to plain text if needed (`return_json=False`)
- Automatic error handling with fallback mechanisms
- Validates JSON structure before returning
- Extracts JSON from markdown code blocks if LLM wraps response

### 3. **New Helper Methods**

#### `_parse_json_response(response_text: str) -> Dict`
- Parses JSON from LLM response
- Handles markdown code blocks

#### `_extract_json_from_text(text: str) -> Optional[str]`
- Extracts JSON from various formats:
  - ```json code blocks
  - ``` generic code blocks
  - Direct JSON objects
- Uses regex patterns to find JSON

#### `_validate_response_structure(response: Dict) -> bool`
- Validates required keys: `greeting`, `ingredients`, `steps`, `closing`
- Checks data types (lists, dicts)
- Validates ingredient objects have `text` field
- Validates step objects have `step_num` and `text` fields

#### `_fallback_to_structured_response(...) -> Dict`
- Creates structured response from raw recipe data
- Used when LLM fails to return valid JSON
- Manually constructs the expected structure

#### `_convert_to_plain_text(response_text: str) -> str`
- Removes markdown formatting
- Returns clean plain text

### 4. **New Utility Methods**

#### `structured_to_plain_text(structured_response: Dict) -> str`
- Converts structured JSON to plain text for TTS
- Concatenates: greeting → ingredients → steps → closing
- Adds natural transitions ("For ingredients, you'll need:", "Now for the cooking steps.")

#### `generate_conversational_response(...) -> str`
- Generates contextual responses based on intent
- Uses conversation history (last 6 turns)
- Uses session context (recipe, step, paused state)
- Returns 1-3 sentence responses suitable for TTS

#### `answer_recipe_question(...) -> str`
- Answers questions using RAG context
- Incorporates conversation history
- Provides general cooking knowledge if answer not in context
- 2-4 sentence responses optimized for TTS

## Benefits

### 1. **Precise Tracking**
- Know exactly which ingredients have been spoken
- Track step-by-step progress
- Can resume from exact position after interruption

### 2. **Better TTS Management**
- Split content into logical chunks (greeting, ingredients, steps, closing)
- Each ingredient/step is 1-2 sentences (optimal for TTS chunking)
- Can speak individual sections on demand

### 3. **Session State Integration**
- JSON structure aligns with SessionManager expectations
- Easy to store in Redis with `response_structure` field
- Can mark items as spoken: `{"text": "...", "spoken": true}`

### 4. **Flexible Output**
- Structured JSON for programmatic use
- Plain text fallback for simple scenarios
- Conversion methods for different use cases

### 5. **Robust Error Handling**
- Multiple fallback layers
- JSON extraction from various formats
- Validation before use
- Manual structure creation if all else fails

## Usage Examples

### Basic Usage
```python
from modules.llm import RecipeLLM

llm = RecipeLLM()

# Get structured response
structured = llm.generate_recipe_response(
    user_query="How do I make pasta?",
    recipe_results=recipe_data,
    return_json=True
)

# Access sections
print(structured['greeting'])
for ing in structured['ingredients']:
    print(ing['text'])
for step in structured['steps']:
    print(f"Step {step['step_num']}: {step['text']}")
print(structured['closing'])
```

### Convert to Plain Text for TTS
```python
plain_text = llm.structured_to_plain_text(structured)
tts.generate_speech(plain_text)
```

### Conversational Response
```python
response = llm.generate_conversational_response(
    user_input="What's next?",
    intent="nav_next",
    conversation_history=history,
    context=session_context
)
```

### Answer Questions
```python
answer = llm.answer_recipe_question(
    question="Can I use butter instead of oil?",
    recipe_context=retrieved_chunks,
    conversation_history=history
)
```

## Integration with Other Components

### Session Manager
```python
session_data = {
    "response_structure": structured,  # Store full JSON
    "ingredients_spoken": [0, 1],      # Indices of spoken ingredients
    "steps_spoken": [1],                # Step numbers spoken
    "current_section": "ingredients"
}
```

### TTS Chunking
```python
# Speak greeting
tts.generate_speech(structured['greeting'])

# Speak ingredients one by one
for i, ing in enumerate(structured['ingredients']):
    tts.generate_speech(ing['text'])
    session_manager.mark_ingredient_spoken(session_id, i)

# Speak steps one by one
for step in structured['steps']:
    tts.generate_speech(step['text'])
    session_manager.mark_step_spoken(session_id, step['step_num'])

# Speak closing
tts.generate_speech(structured['closing'])
```

### Navigator Integration
```python
# Get current position
current_section = session_data['current_section']  # "ingredients" or "steps"

if intent == "nav_next":
    if current_section == "ingredients":
        next_ing = get_next_unspoken_ingredient(structured)
        if next_ing:
            speak(next_ing['text'])
        else:
            # Move to steps
            session_data['current_section'] = "steps"
    elif current_section == "steps":
        next_step = get_next_unspoken_step(structured)
        if next_step:
            speak(next_step['text'])
```

## Testing

Run the test script:
```bash
python test_llm_structured.py
```

Tests cover:
1. Structured JSON generation
2. Structure validation
3. Plain text conversion
4. No results handling
5. Conversational responses
6. Question answering

## Next Steps

1. **Integrate with TTS chunking** - Split sentences and generate audio chunks
2. **Implement Navigator** - Use structured data for navigation commands
3. **Connect to Session Manager** - Store and track progress
4. **Add interruption handling** - Resume from exact position
5. **Implement RAG for questions** - Use retriever for context

## Model Configuration

Current model: `gemini-2.0-flash-exp`
- Fast response times
- Good JSON generation
- Conversational tone
- Can be changed in `__init__` if needed

## Error Handling

The implementation has 3 layers of fallback:
1. **Primary**: LLM generates valid JSON
2. **Secondary**: Extract JSON from markdown/text
3. **Tertiary**: Manually construct structure from raw data

This ensures the system never fails completely, even if LLM returns unexpected format.

## Prompt Engineering

The prompt explicitly requests:
- Valid JSON only (no markdown)
- Specific structure with all fields
- 1-2 sentences per ingredient/step (optimal for TTS)
- Conversational, spoken language
- Natural tone for listening while cooking

## Validation Rules

- All required keys must exist: `greeting`, `ingredients`, `steps`, `closing`
- `ingredients` must be list of dicts with `text` field
- `steps` must be list of dicts with `step_num` and `text` fields
- Strings must be non-empty
- Step numbers should be sequential (not enforced, but expected)

---

**Status**: ✅ **COMPLETE**

The LLM module now fully supports structured JSON output with robust error handling, conversation history, and utility methods for integration with other components.

