import os
import json
import re
from google import genai
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union


class RecipeLLM:
    def __init__(self, model_name="gemini-2.5-flash"):
        """
        Initialize the Recipe LLM using Google Gemini
        
        Args:
            model_name (str): The Gemini model to use
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv("Gemini_API_key")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
    def generate_recipe_response(self, user_query: str, recipe_results: List[Dict],
                                 return_json: bool = True) -> Union[Dict, str]:
        """
        Generate a structured JSON response with recipe details for TTS and tracking

        Args:
            user_query (str): The original user query/transcription
            recipe_results (list): List of recipe results from the retriever
            return_json (bool): If True, return structured JSON. If False, return plain text (fallback)

        Returns:
            Dict or str: Structured JSON response or plain text fallback

        JSON Structure:
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
        """
        try:
            # Handle case with no results
            if not recipe_results or len(recipe_results) == 0:
                if return_json:
                    return {
                        "greeting": "I couldn't find any recipes matching your request.",
                        "ingredients": [],
                        "steps": [],
                        "closing": "Please try searching for a different recipe."
                    }
                else:
                    return "I couldn't find any recipes matching your request. Please try searching for a different recipe."

            # Use only the top 1 recipe (best match)
            top_recipe = recipe_results[0]
            recipe_title = top_recipe.get('title', 'Unknown')

            # Format ingredients for context
            ingredients_list = []
            if 'ingredients' in top_recipe and top_recipe['ingredients']:
                for ing in top_recipe['ingredients']:
                    ingredient_name = ing.get('ingredient', '')
                    quantity = ing.get('quantity', '')
                    unit = ing.get('unit', '')

                    # Build ingredient string
                    ing_str = ingredient_name
                    if quantity:
                        ing_str = f"{quantity} {unit} {ingredient_name}".strip()

                    if ing_str:
                        ingredients_list.append(ing_str)

            ingredients_formatted = ", ".join(ingredients_list) if ingredients_list else "No ingredients found"

            # Format instructions for context
            instructions_list = []
            if 'instructions' in top_recipe and top_recipe['instructions']:
                for i, instruction in enumerate(top_recipe['instructions'], 1):
                    if isinstance(instruction, dict):
                        step_text = instruction.get('step', '')
                    else:
                        step_text = str(instruction)

                    if step_text:
                        instructions_list.append(f"Step {i}: {step_text}")

            instructions_formatted = "\n".join(instructions_list) if instructions_list else "No instructions found"

            recipes_context = f"""Recipe: {recipe_title}

Ingredients: {ingredients_formatted}

Instructions:
{instructions_formatted}"""

            print("Recipes Context for LLM:")
            print(recipes_context)
            print()

            # Build prompt for structured JSON output
            prompt = f"""The user asked: "{user_query}"

Here is the best matching recipe found:

{recipes_context}

IMPORTANT: You must respond ONLY with valid JSON in the following structure. Do not include any markdown formatting, code blocks, or additional text outside the JSON.

{{
  "greeting": "A brief, friendly welcome message (1-2 sentences) mentioning the recipe name",
  "ingredients": [
    {{"text": "First ingredient with quantity in conversational tone"}},
    {{"text": "Second ingredient with quantity in conversational tone"}},
    ...
  ],
  "steps": [
    {{"step_num": 1, "text": "First cooking step in simple, conversational language (1-2 sentences)"}},
    {{"step_num": 2, "text": "Second cooking step in simple, conversational language (1-2 sentences)"}},
    ...
  ],
  "closing": "A brief, encouraging closing message (1 sentence)"
}}

Guidelines:
1. Keep each ingredient text to 1 sentence, conversational and natural for TTS
2. Keep each step text to 1-2 sentences max, easy to follow while cooking
3. Use spoken language, avoid technical jargon
4. Be encouraging and friendly in tone
5. Do NOT include recipe IDs, similarity scores, or metadata
6. Make it sound natural for someone listening while cooking

Return ONLY the JSON object, no additional text or formatting."""

            # Generate response using Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            response_text = response.text.strip()

            # Try to parse JSON response
            if return_json:
                try:
                    structured_response = self._parse_json_response(response_text)

                    # Validate structure
                    if self._validate_response_structure(structured_response):
                        print("✓ Successfully generated structured JSON response")
                        return structured_response
                    else:
                        print("⚠ Invalid response structure, attempting fallback...")
                        return self._fallback_to_structured_response(
                            user_query, recipe_title, ingredients_list, instructions_list
                        )

                except json.JSONDecodeError as e:
                    print(f"⚠ JSON parsing failed: {e}")
                    print("Attempting to extract JSON from response...")

                    # Try to extract JSON from markdown code blocks
                    cleaned_response = self._extract_json_from_text(response_text)
                    if cleaned_response:
                        try:
                            structured_response = json.loads(cleaned_response)
                            if self._validate_response_structure(structured_response):
                                print("✓ Successfully extracted and parsed JSON")
                                return structured_response
                        except:
                            pass

                    # Fallback to structured response
                    print("Using fallback structured response")
                    return self._fallback_to_structured_response(
                        user_query, recipe_title, ingredients_list, instructions_list
                    )
            else:
                # Return plain text
                return self._convert_to_plain_text(response_text)

        except Exception as e:
            print(f"Error generating LLM response: {str(e)}")

            # Return error fallback
            if return_json:
                return {
                    "greeting": "I found some recipes for you, but I'm having trouble describing them right now.",
                    "ingredients": [],
                    "steps": [],
                    "closing": "Please check the display for details."
                }
            else:
                return "I found some recipes for you, but I'm having trouble describing them right now. Please check the display for details."

    def _parse_json_response(self, response_text: str) -> Dict:
        """
        Parse JSON response from LLM

        Args:
            response_text (str): Raw response text from LLM

        Returns:
            Dict: Parsed JSON object
        """
        # Remove markdown code blocks if present
        cleaned = self._extract_json_from_text(response_text)
        if not cleaned:
            cleaned = response_text

        return json.loads(cleaned)

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON from text that may contain markdown code blocks

        Args:
            text (str): Text potentially containing JSON

        Returns:
            str or None: Extracted JSON string or None
        """
        # Try to extract from ```json code blocks
        json_match = re.search(r'```json\s*(\{.*?})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Try to extract from ``` code blocks
        code_match = re.search(r'```\s*(\{.*?})\s*```', text, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # Try to find JSON object directly
        json_obj_match = re.search(r'\{.*}', text, re.DOTALL)
        if json_obj_match:
            return json_obj_match.group(0)

        return None

    def _validate_response_structure(self, response: Dict) -> bool:
        """
        Validate that response has required structure

        Args:
            response (dict): Response to validate

        Returns:
            bool: True if valid, False otherwise
        """
        required_keys = ["greeting", "ingredients", "steps", "closing"]

        # Check all required keys exist
        if not all(key in response for key in required_keys):
            return False

        # Check ingredients is a list
        if not isinstance(response["ingredients"], list):
            return False

        # Check steps is a list
        if not isinstance(response["steps"], list):
            return False

        # Check each ingredient has "text" field
        for ing in response["ingredients"]:
            if not isinstance(ing, dict) or "text" not in ing:
                return False

        # Check each step has "step_num" and "text" fields
        for step in response["steps"]:
            if not isinstance(step, dict) or "step_num" not in step or "text" not in step:
                return False

        return True

    def _fallback_to_structured_response(self, user_query: str, recipe_title: str,
                                        ingredients_list: List[str],
                                        instructions_list: List[str]) -> Dict:
        """
        Create structured response from raw recipe data (fallback)

        Args:
            user_query (str): User's query
            recipe_title (str): Recipe title
            ingredients_list (list): List of ingredient strings
            instructions_list (list): List of instruction strings

        Returns:
            Dict: Structured response
        """
        # Create structured response manually
        structured = {
            "greeting": f"Great! Let me help you make {recipe_title}.",
            "ingredients": [
                {"text": ing, "spoken": False}
                for ing in ingredients_list
            ],
            "steps": [],
            "closing": "Enjoy your cooking!"
        }

        # Parse steps from instructions
        for instruction in instructions_list:
            # Extract step number and text
            match = re.match(r'Step (\d+):\s*(.+)', instruction)
            if match:
                step_num = int(match.group(1))
                step_text = match.group(2)
                structured["steps"].append({
                    "step_num": step_num,
                    "text": step_text,
                    "spoken": False
                })

        return structured

    def _convert_to_plain_text(self, response_text: str) -> str:
        """
        Convert response to plain text (remove markdown, etc.)

        Args:
            response_text (str): Response text

        Returns:
            str: Plain text version
        """
        # Remove markdown code blocks
        text = re.sub(r'```[a-z]*\n', '', response_text)
        text = re.sub(r'```', '', text)

        return text.strip()

    def structured_to_plain_text(self, structured_response: Dict) -> str:
        """
        Convert structured JSON response to plain text for TTS

        Args:
            structured_response (dict): Structured response with greeting, ingredients, steps, closing

        Returns:
            str: Plain text version suitable for TTS
        """
        parts = []

        # Add greeting
        if structured_response.get("greeting"):
            parts.append(structured_response["greeting"])

        # Add ingredients
        ingredients = structured_response.get("ingredients", [])
        if ingredients:
            parts.append("For ingredients, you'll need:")
            for ing in ingredients:
                parts.append(ing.get("text", ""))

        # Add steps
        steps = structured_response.get("steps", [])
        if steps:
            parts.append("Now for the cooking steps.")
            for step in steps:
                step_text = step.get("text", "")
                parts.append(step_text)

        # Add closing
        if structured_response.get("closing"):
            parts.append(structured_response["closing"])

        return " ".join(parts)

    def generate_conversational_response(self, user_input: str, intent: str,
                                        conversation_history: Optional[List[Dict]] = None,
                                        context: Optional[Dict] = None) -> str:
        """
        Generate conversational response based on intent and context

        Args:
            user_input (str): User's input text
            intent (str): Detected intent
            conversation_history (list, optional): Previous conversation turns
            context (dict, optional): Session context (current recipe, step, etc.)

        Returns:
            str: Generated response
        """
        try:
            # Build conversation history string
            history_str = ""
            if conversation_history:
                recent_history = conversation_history[-6:]  # Last 3 turns (user + assistant)
                for turn in recent_history:
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    history_str += f"{role.capitalize()}: {content}\n"

            # Build context string
            context_str = ""
            if context:
                recipe_title = context.get("recipe_title", "")
                step_index = context.get("step_index", 0)
                current_section = context.get("current_section", "")
                paused = context.get("paused", False)

                context_str = f"""
Current Context:
- Recipe: {recipe_title if recipe_title else 'None'}
- Current Step: {step_index}
- Section: {current_section}
- Paused: {paused}
"""

            # Create prompt based on intent
            prompt = f"""You are a helpful cooking assistant having a conversation with a user.

{context_str}

Recent Conversation:
{history_str if history_str else "No previous conversation"}

User Intent: {intent}
User Input: "{user_input}"

Generate a brief, friendly, conversational response (1-3 sentences) appropriate for the intent and context.
Keep it natural for text-to-speech. Be helpful and encouraging.

Response:"""

            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            return response.text.strip()

        except Exception as e:
            print(f"Error generating conversational response: {e}")
            return "I'm here to help! What would you like to know?"

    def answer_recipe_question(self, question: str, recipe_context: str,
                               conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Answer a question about the recipe using RAG context

        Args:
            question (str): User's question
            recipe_context (str): Retrieved recipe context/chunks
            conversation_history (list, optional): Previous conversation

        Returns:
            str: Answer to the question
        """
        try:
            # Build history
            history_str = ""
            if conversation_history:
                recent_history = conversation_history[-4:]
                for turn in recent_history:
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    history_str += f"{role.capitalize()}: {content}\n"

            # Build conversation section
            conversation_section = f"Recent Conversation:\n{history_str}" if history_str else ""

            prompt = f"""You are a knowledgeable cooking assistant. Answer the user's question based on the recipe context provided.

Recipe Context:
{recipe_context}

{conversation_section}

User Question: "{question}"

Provide a clear, concise answer (2-4 sentences) in conversational language suitable for text-to-speech.
If the answer isn't in the recipe context, provide general cooking knowledge but mention you're making an educated guess.

Answer:"""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            return response.text.strip()

        except Exception as e:
            print(f"Error answering question: {e}")
            return "I'm not sure about that. Could you rephrase your question?"

    def generate_simple_response(self, prompt):
        """
        Generate a simple response for any prompt
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            str: The generated response
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm having trouble generating a response right now."
