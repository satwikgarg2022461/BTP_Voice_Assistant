import os
from google import genai
from dotenv import load_dotenv


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
        
    def generate_recipe_response(self, user_query, recipe_results):
        """
        Generate a natural language response with detailed recipe ingredients and steps for TTS
        
        Args:
            user_query (str): The original user query/transcription
            recipe_results (list): List of recipe results from the retriever
            
        Returns:
            str: Natural language response with recipe details suitable for TTS
        """
        try:
            # Handle case with no results
            if not recipe_results or len(recipe_results) == 0:
                prompt = f"""The user asked: "{user_query}"

Unfortunately, no matching recipes were found in the database. 

Generate a friendly, conversational response (2-3 sentences) telling the user that no recipes were found and suggesting they try a different query. Make it sound natural for text-to-speech."""
            else:
                # Use only the top 1 recipe (best match)
                top_recipe = recipe_results[0]
                recipe_title = top_recipe.get('title', 'Unknown')
                
                # Format ingredients
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
                
                # Format instructions
                instructions_list = []
                if 'instructions' in top_recipe and top_recipe['instructions']:
                    for i, instruction in enumerate(top_recipe['instructions'], 1):
                        if isinstance(instruction, dict):
                            step_text = instruction.get('step', '')
                        else:
                            step_text = str(instruction)
                        
                        if step_text:
                            instructions_list.append(f"Step {i}: {step_text}")
                
                instructions_formatted = " ".join(instructions_list) if instructions_list else "No instructions found"
                
                recipes_context = f"""Recipe: {recipe_title}

Ingredients: {ingredients_formatted}

Instructions: {instructions_formatted}"""
                
                print("Recipes Context for LLM:")
                print(recipes_context)
                print()
                
                prompt = f"""The user asked: "{user_query}"

Here is the best matching recipe found:

{recipes_context}

Generate a detailed, friendly response (8-12 sentences) that:
1. Warmly greet what the user asked for and mention the recipe name
2. Extract and read out the INGREDIENTS section naturally - list each ingredient with quantities in a conversational way (e.g., "You'll need one cup of flour, two tablespoons of butter, and so on")
3. Extract and read out the COOKING INSTRUCTIONS/STEPS section - describe the cooking process step-by-step in simple, easy-to-follow language as if you're speaking to someone who will be cooking
4. Keep the tone friendly, encouraging, and spoken - avoid technical jargon and make it sound natural for text-to-speech output
5. Organize ingredients first, then instructions, in a logical order
6. Make it conversational and easy to follow while cooking
7. Make it short and concise.

Do NOT mention recipe IDs, similarity scores, or metadata. Just focus on ingredients and cooking instructions in a natural, spoken tone suitable for someone listening while cooking."""

            # Generate response using Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating LLM response: {str(e)}")
            return "I found some recipes for you, but I'm having trouble describing them right now. Please check the display for details."
    
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
