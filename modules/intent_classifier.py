import re
import os
from enum import Enum
from typing import Dict, Tuple, Optional, List
from google import genai
from dotenv import load_dotenv


class Intent(Enum):
    """Enumeration of all possible user intents"""
    NAV_NEXT = "nav_next"
    NAV_PREV = "nav_prev"
    NAV_GO_TO = "nav_go_to"
    NAV_REPEAT = "nav_repeat"
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


class IntentClassifier:
    """
    Hybrid Intent Classifier using rule-based patterns with LLM fallback

    Strategy:
    1. Try rule-based classification with confidence score
    2. If confidence < threshold, use LLM for classification
    3. Return intent + confidence + extracted entities
    """

    def __init__(self, confidence_threshold=0.7, use_llm_fallback=True):
        """
        Initialize the Intent Classifier

        Args:
            confidence_threshold (float): Minimum confidence for rule-based classification
            use_llm_fallback (bool): Whether to use LLM when rule-based confidence is low
        """
        load_dotenv()

        self.confidence_threshold = confidence_threshold
        self.use_llm_fallback = use_llm_fallback

        # Initialize Gemini client for LLM fallback
        if use_llm_fallback:
            api_key = os.getenv("Gemini_API_key")
            if api_key:
                self.client = genai.Client(api_key=api_key)
                self.model_name = "gemini-2.5-flash"
                print("Intent Classifier initialized with LLM fallback")
            else:
                print("Warning: Gemini API key not found. LLM fallback disabled.")
                self.use_llm_fallback = False

        # Define rule-based patterns for each intent
        self.intent_patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[Intent, List[Dict]]:
        """
        Initialize regex patterns and keywords for rule-based classification

        Returns:
            Dictionary mapping intents to their patterns and keywords
        """
        return {
            Intent.NAV_NEXT: [
                {"pattern": r"\b(next|continue|forward|proceed|go ahead|move on)\b", "confidence": 0.95},
                {"pattern": r"\b(what'?s next|after that|then what)\b", "confidence": 0.9},
                {"pattern": r"\b(skip|move forward)\b", "confidence": 0.85},
            ],
            Intent.NAV_PREV: [
                {"pattern": r"\b(previous|back|before|earlier|go back|last step)\b", "confidence": 0.95},
                {"pattern": r"\b(what was (that|the last)|can you repeat|say that again)\b", "confidence": 0.85},
                {"pattern": r"\b(undo|rewind)\b", "confidence": 0.8},
            ],
            Intent.NAV_GO_TO: [
                {"pattern": r"\b(go to|jump to|skip to) (step|ingredient)?\s*(\d+|first|last|beginning|end)\b", "confidence": 0.95},
                {"pattern": r"\b(step|ingredient)?\s*(\d+|first|last)\b", "confidence": 0.7},
            ],
            Intent.NAV_START: [
                {"pattern": r"\b(repeat|start|begin).*(from )?(the )?(beginning|start|top|starting)\b", "confidence": 0.95},
                {"pattern": r"\b(start|begin|let'?s (start|begin|go)|show me how)\b", "confidence": 0.95},
                {"pattern": r"\b(from the (beginning|start|top)|take me through)\b", "confidence": 0.9},
                {"pattern": r"\b(restart|start over|begin again)\b", "confidence": 0.95},
            ],
            Intent.NAV_REPEAT: [
                {"pattern": r"\b(repeat|say (that|it) again|one more time|again|pardon)(?!.*(beginning|start|top|starting))\b", "confidence": 0.95},
                {"pattern": r"\b(what did you say|didn'?t (catch|hear) that)\b", "confidence": 0.9},
                {"pattern": r"\b(come again|excuse me)\b", "confidence": 0.75},
            ],
            Intent.SMALL_TALK: [
                {"pattern": r"\b(hi|hello|hey|good (morning|afternoon|evening)|greetings)\b", "confidence": 0.95},
                {"pattern": r"\b(how are you|how'?re you|how are u|how r u|how'?s it going|what'?s up)\b", "confidence": 0.95},
                {"pattern": r"\b(thanks|thank you|bye|goodbye|see you|take care)\b", "confidence": 0.95},
                {"pattern": r"\b(nice|great|awesome|cool|good job|well done)\b$", "confidence": 0.9},
                {"pattern": r"\b(weather|how'?s your day|doing today|feeling)\b", "confidence": 0.85},
            ],
            Intent.QUESTION: [
                {"pattern": r"\b(substitute|replace|alternative|instead of|swap)\b", "confidence": 0.9},
                {"pattern": r"\b(how much|how many|how long|what temperature|what time)\b", "confidence": 0.95},
                {"pattern": r"\b(why|when|where|which)\b.*(step|ingredient|recipe|cook|add|mix|heat)\b", "confidence": 0.85},
                {"pattern": r"\b(what|how).*(temperature|time|long|much|many|ingredient|substitute)\b", "confidence": 0.85},
                {"pattern": r"\b(can i|could i|should i|is it okay).*(use|add|replace|skip)\b", "confidence": 0.9},
                {"pattern": r"\b(tell me (about|more)|explain|describe).*(recipe|step|ingredient|process)\b", "confidence": 0.85},
            ],
            Intent.SEARCH_RECIPE: [
                {"pattern": r"\b(find|search|look for|show me|give me|i want|i need) (a |some |the )?(recipe|dish)\b", "confidence": 0.95},
                {"pattern": r"\b(how (do|to) (make|cook|prepare)|recipe for)\b", "confidence": 0.9},
                {"pattern": r"\b(cook|make|prepare)\s+\w+", "confidence": 0.7},
            ],
            Intent.START_RECIPE: [
                {"pattern": r"\b(start|begin|let'?s (do|make|cook)) (this|that|it|the recipe)\b", "confidence": 0.95},
                {"pattern": r"\b(okay let'?s go|let'?s start cooking|ready to cook)\b", "confidence": 0.9},
                {"pattern": r"\b(show me (the |how to )?steps|walk me through)\b", "confidence": 0.85},
            ],
            Intent.STOP_PAUSE: [
                {"pattern": r"\b(stop|pause|wait|hold on|hang on)\b", "confidence": 0.95},
                {"pattern": r"\b(just a (second|minute|moment)|give me a (second|minute))\b", "confidence": 0.9},
                {"pattern": r"\b(cancel|never mind|stop reading)\b", "confidence": 0.85},
            ],
            Intent.RESUME: [
                {"pattern": r"\b(resume|continue|go on|keep going|carry on)\b", "confidence": 0.95},
                {"pattern": r"\b(okay (continue|go ahead)|i'?m back|ready now)\b", "confidence": 0.9},
            ],
            Intent.CONFIRM: [
                {"pattern": r"\b(yes|yeah|yep|sure|okay|ok|alright|correct|right|exactly)\b$", "confidence": 0.95},
                {"pattern": r"\b(that'?s (right|correct)|sounds good|go ahead)\b", "confidence": 0.9},
                {"pattern": r"\b(affirmative|indeed|absolutely)\b", "confidence": 0.85},
            ],
            Intent.CANCEL: [
                {"pattern": r"\b(no|nope|nah|cancel|stop|don'?t|never mind)\b", "confidence": 0.95},
                {"pattern": r"\b(not (really|now)|maybe later|skip (it|this))\b", "confidence": 0.85},
            ],
            Intent.HELP: [
                {"pattern": r"\b(help|assist|support|what can you do|commands)\b", "confidence": 0.95},
                {"pattern": r"\b(how (do|does) (this|it) work|instructions|guide)\b", "confidence": 0.9},
                {"pattern": r"\b(i'?m (lost|confused|stuck)|don'?t understand)\b", "confidence": 0.85},
            ],
        }

    def classify(self, user_input: str, context: Optional[Dict] = None) -> Tuple[Intent, float, Dict]:
        """
        Classify user intent using hybrid approach

        Args:
            user_input (str): The user's spoken/text input
            context (dict, optional): Session context for better classification

        Returns:
            Tuple of (Intent, confidence_score, extracted_entities)
        """
        if not user_input or not user_input.strip():
            return Intent.UNKNOWN, 0.0, {}

        # Normalize input
        normalized_input = user_input.lower().strip()

        # Step 1: Try rule-based classification
        intent, confidence, entities = self._rule_based_classify(normalized_input, context)

        print(f"Rule-based: Intent={intent.value}, Confidence={confidence:.2f}")

        # Step 2: Use LLM fallback if confidence is low
        if confidence < self.confidence_threshold and self.use_llm_fallback:
            print(f"Confidence below threshold ({self.confidence_threshold}), using LLM fallback...")
            llm_intent, llm_confidence, llm_entities = self._llm_classify(user_input, context)

            # Use LLM result if it has higher confidence
            if llm_confidence > confidence:
                print(f"LLM: Intent={llm_intent.value}, Confidence={llm_confidence:.2f} (selected)")
                return llm_intent, llm_confidence, llm_entities
            else:
                print(f"LLM: Intent={llm_intent.value}, Confidence={llm_confidence:.2f} (rule-based kept)")

        return intent, confidence, entities

    def _rule_based_classify(self, normalized_input: str, context: Optional[Dict]) -> Tuple[Intent, float, Dict]:
        """
        Perform rule-based classification using regex patterns

        Args:
            normalized_input (str): Normalized user input
            context (dict, optional): Session context

        Returns:
            Tuple of (Intent, confidence_score, extracted_entities)
        """
        best_match = (Intent.UNKNOWN, 0.0, {})

        # Check each intent's patterns
        for intent, patterns in self.intent_patterns.items():
            for pattern_dict in patterns:
                pattern = pattern_dict["pattern"]
                base_confidence = pattern_dict["confidence"]

                match = re.search(pattern, normalized_input, re.IGNORECASE)
                if match:
                    # Extract entities based on intent type
                    entities = self._extract_entities(intent, match, normalized_input)

                    # Apply context boosting
                    adjusted_confidence = self._apply_context_boost(
                        intent, base_confidence, context, entities
                    )

                    # Keep best match
                    if adjusted_confidence > best_match[1]:
                        best_match = (intent, adjusted_confidence, entities)

        return best_match

    def _extract_entities(self, intent: Intent, match: re.Match, text: str) -> Dict:
        """
        Extract relevant entities based on intent type

        Args:
            intent (Intent): Detected intent
            match (re.Match): Regex match object
            text (str): Full input text

        Returns:
            Dictionary of extracted entities
        """
        entities = {}

        # Extract step numbers for navigation
        if intent in [Intent.NAV_GO_TO]:
            step_match = re.search(r'\b(\d+)\b', text)
            if step_match:
                entities["step_number"] = int(step_match.group(1))
            elif "first" in text or "beginning" in text or "start" in text:
                entities["step_number"] = 1
                entities["position"] = "first"
            elif "last" in text or "end" in text:
                entities["position"] = "last"

        # Extract recipe name for search
        if intent == Intent.SEARCH_RECIPE:
            # Try to extract recipe name after trigger words
            recipe_patterns = [
                r'(?:recipe for|make|cook|prepare|find|search for|show me)\s+(?:a |an |the |some )?(.+?)(?:\?|$)',
                r'(?:how to make|how to cook|how to prepare)\s+(?:a |an |the |some )?(.+?)(?:\?|$)',
            ]
            for pattern in recipe_patterns:
                recipe_match = re.search(pattern, text, re.IGNORECASE)
                if recipe_match:
                    entities["recipe_name"] = recipe_match.group(1).strip()
                    break

        # Extract question content
        if intent == Intent.QUESTION:
            entities["question_text"] = text
            # Identify question type
            if re.search(r'\b(substitute|replace|instead of|alternative)\b', text):
                entities["question_type"] = "substitution"
            elif re.search(r'\b(how long|how much time|duration)\b', text):
                entities["question_type"] = "timing"
            elif re.search(r'\b(how much|how many|quantity)\b', text):
                entities["question_type"] = "quantity"
            elif re.search(r'\b(what temperature|how hot)\b', text):
                entities["question_type"] = "temperature"
            else:
                entities["question_type"] = "general"

        return entities

    def _apply_context_boost(self, intent: Intent, base_confidence: float,
                            context: Optional[Dict], entities: Dict) -> float:
        """
        Adjust confidence based on session context

        Args:
            intent (Intent): Detected intent
            base_confidence (float): Base confidence from pattern matching
            context (dict, optional): Session context
            entities (dict): Extracted entities

        Returns:
            Adjusted confidence score
        """
        if not context:
            return base_confidence

        confidence = base_confidence
        current_state = context.get("current_state", "IDLE")

        # Boost navigation intents when in active recipe
        if intent in [Intent.NAV_NEXT, Intent.NAV_PREV, Intent.NAV_REPEAT, Intent.NAV_GO_TO]:
            if current_state in ["READING_INGREDIENTS", "READING_STEPS", "RECIPE_ACTIVE"]:
                confidence = min(1.0, confidence + 0.1)

        # Boost START_RECIPE when recipe is selected
        if intent == Intent.START_RECIPE:
            if current_state == "RECIPE_SELECTED":
                confidence = min(1.0, confidence + 0.15)

        # Boost RESUME when paused
        if intent == Intent.RESUME:
            if context.get("paused") == "true" or current_state == "PAUSED":
                confidence = min(1.0, confidence + 0.15)

        # Boost QUESTION when in active recipe
        if intent == Intent.QUESTION:
            if current_state in ["READING_INGREDIENTS", "READING_STEPS", "RECIPE_ACTIVE"]:
                confidence = min(1.0, confidence + 0.05)

        return confidence

    def _llm_classify(self, user_input: str, context: Optional[Dict]) -> Tuple[Intent, float, Dict]:
        """
        Use LLM to classify ambiguous intents

        Args:
            user_input (str): The user's input
            context (dict, optional): Session context

        Returns:
            Tuple of (Intent, confidence_score, extracted_entities)
        """
        try:
            # Build context string
            context_str = ""
            if context:
                context_str = f"""
Current Session Context:
- State: {context.get('current_state', 'IDLE')}
- Recipe Active: {context.get('recipe_id', 'None')}
- Current Section: {context.get('current_section', 'None')}
- Paused: {context.get('paused', 'false')}
"""

            # Create prompt for LLM
            prompt = f"""You are an intent classifier for a conversational recipe voice assistant. 
Classify the user's input into ONE of the following intents:

Available Intents:
- nav_next: User wants to go to the next step/ingredient
- nav_prev: User wants to go back to previous step/ingredient
- nav_go_to: User wants to jump to a specific step/ingredient
- nav_repeat: User wants to hear the current step again
- nav_start: User wants to start from the beginning
- question: User is asking a question about the recipe
- search_recipe: User wants to find/search for a recipe
- start_recipe: User wants to start cooking a selected recipe
- stop_pause: User wants to pause or stop
- resume: User wants to resume after pausing
- confirm: User is confirming/agreeing (yes, okay, etc.)
- cancel: User is canceling/disagreeing (no, cancel, etc.)
- small_talk: Greetings or casual conversation or weather info 
- clarify: User needs clarification
- help: User needs help or instructions
- unknown: Cannot determine intent

{context_str}

User Input: "{user_input}"

Respond in JSON format:
{{
    "intent": "intent_name",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "entities": {{}}
}}

Extract relevant entities (e.g., step numbers, recipe names, question text) in the entities field.
"""

            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            # Parse response
            response_text = response.text.strip()
            print(f"LLM Response: {response_text}")

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            import json
            result = json.loads(response_text)

            # Map string intent to Intent enum
            intent_str = result.get("intent", "unknown")
            try:
                intent = Intent(intent_str)
            except ValueError:
                intent = Intent.UNKNOWN

            confidence = float(result.get("confidence", 0.5))
            entities = result.get("entities", {})

            return intent, confidence, entities

        except Exception as e:
            print(f"Error in LLM classification: {str(e)}")
            return Intent.UNKNOWN, 0.3, {}

    def get_intent_description(self, intent: Intent) -> str:
        """
        Get human-readable description of an intent

        Args:
            intent (Intent): The intent to describe

        Returns:
            Description string
        """
        descriptions = {
            Intent.NAV_NEXT: "Navigate to next step/ingredient",
            Intent.NAV_PREV: "Navigate to previous step/ingredient",
            Intent.NAV_GO_TO: "Jump to specific step/ingredient",
            Intent.NAV_REPEAT: "Repeat current step",
            Intent.NAV_START: "Start from beginning",
            Intent.QUESTION: "Ask a question about the recipe",
            Intent.SEARCH_RECIPE: "Search for a recipe",
            Intent.START_RECIPE: "Begin cooking selected recipe",
            Intent.STOP_PAUSE: "Pause or stop",
            Intent.RESUME: "Resume from pause",
            Intent.CONFIRM: "Confirm/agree",
            Intent.CANCEL: "Cancel/disagree",
            Intent.SMALL_TALK: "Casual conversation",
            Intent.CLARIFY: "Request clarification",
            Intent.HELP: "Request help",
            Intent.UNKNOWN: "Cannot determine intent"
        }
        return descriptions.get(intent, "Unknown intent")


# Test function
def test_intent_classifier():
    """Test the intent classifier with sample inputs"""
    classifier = IntentClassifier(confidence_threshold=0.7, use_llm_fallback=True)

    test_cases = [
        ("next step please", None),
        ("go back", None),
        ("what's the third step?", {"current_state": "RECIPE_ACTIVE"}),
        ("repeat that", {"current_state": "READING_STEPS"}),
        ("how do I make pasta?", None),
        ("start cooking", {"current_state": "RECIPE_SELECTED"}),
        ("pause", None),
        ("can I use butter instead of oil?", {"current_state": "RECIPE_ACTIVE"}),
        ("yes", None),
        ("what is weather today", None),
        ("Can u repeat the recipe from starting?", None),
        ("well how are u doing today?", None),
        ("find me a recipe for chocolate cake", None),
        ("let's start cooking this recipe", {"current_state": "RECIPE_SELECTED"}),
        ("resume where we left off", {"current_state": "PAUSED"}),
        ("no, I don't want that", None),
        ("help me with the commands", None),
        ("", None),
        ("   ", None),
        ("blablabla unknown input", None),
        ("how to make a vegan salad?", None),
    ]

    print("=" * 60)
    print("INTENT CLASSIFIER TEST")
    print("=" * 60)

    for user_input, context in test_cases:
        print(f"\nInput: '{user_input}'")
        if context:
            print(f"Context: {context}")

        intent, confidence, entities = classifier.classify(user_input, context)

        print(f"→ Intent: {intent.value}")
        print(f"→ Confidence: {confidence:.2f}")
        print(f"→ Description: {classifier.get_intent_description(intent)}")
        if entities:
            print(f"→ Entities: {entities}")
        print("-" * 60)


if __name__ == "__main__":
    test_intent_classifier()

