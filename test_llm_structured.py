#!/usr/bin/env python3
"""
Test script for structured JSON LLM responses
"""

import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.llm import RecipeLLM


def test_structured_response():
    """Test structured JSON response generation"""
    print("=" * 70)
    print("TESTING STRUCTURED JSON LLM RESPONSE")
    print("=" * 70)

    # Initialize LLM
    print("\n1. Initializing LLM...")
    llm = RecipeLLM()
    print("✓ LLM initialized")

    # Mock recipe data
    mock_recipe_results = [
        {
            'recipe_id': '123',
            'title': 'Simple Spaghetti Carbonara',
            'ingredients': [
                {'ingredient': 'spaghetti', 'quantity': '400', 'unit': 'g'},
                {'ingredient': 'eggs', 'quantity': '4', 'unit': ''},
                {'ingredient': 'parmesan cheese', 'quantity': '100', 'unit': 'g'},
                {'ingredient': 'pancetta', 'quantity': '200', 'unit': 'g'},
                {'ingredient': 'black pepper', 'quantity': '1', 'unit': 'tsp'}
            ],
            'instructions': [
                {'step': 'Bring a large pot of salted water to boil and cook spaghetti according to package directions.'},
                {'step': 'Meanwhile, beat eggs in a bowl and mix in grated parmesan cheese.'},
                {'step': 'Cook diced pancetta in a large pan until crispy.'},
                {'step': 'Drain pasta, reserving 1 cup of pasta water.'},
                {'step': 'Add hot pasta to the pancetta pan, remove from heat, then quickly mix in the egg mixture, adding pasta water as needed to create a creamy sauce.'},
                {'step': 'Season with black pepper and serve immediately.'}
            ]
        }
    ]

    # Test 1: Generate structured JSON response
    print("\n2. Generating structured JSON response...")
    user_query = "How do I make spaghetti carbonara?"

    structured_response = llm.generate_recipe_response(
        user_query=user_query,
        recipe_results=mock_recipe_results,
        return_json=True
    )

    print("\n✓ Structured Response Generated:")
    print(json.dumps(structured_response, indent=2))

    # Validate structure
    print("\n3. Validating response structure...")
    required_keys = ['greeting', 'ingredients', 'steps', 'closing']
    for key in required_keys:
        if key in structured_response:
            print(f"  ✓ '{key}' present")
        else:
            print(f"  ✗ '{key}' MISSING")

    # Check ingredients format
    ingredients = structured_response.get('ingredients', [])
    print(f"\n  Ingredients count: {len(ingredients)}")
    if ingredients:
        print(f"  First ingredient: {ingredients[0]}")
        if 'text' in ingredients[0]:
            print("  ✓ Ingredients have 'text' field")

    # Check steps format
    steps = structured_response.get('steps', [])
    print(f"\n  Steps count: {len(steps)}")
    if steps:
        print(f"  First step: {steps[0]}")
        if 'step_num' in steps[0] and 'text' in steps[0]:
            print("  ✓ Steps have 'step_num' and 'text' fields")

    # Test 2: Convert to plain text
    print("\n4. Converting to plain text for TTS...")
    plain_text = llm.structured_to_plain_text(structured_response)
    print("\n✓ Plain Text Version:")
    print("-" * 70)
    print(plain_text)
    print("-" * 70)

    # Test 3: Test with no results
    print("\n5. Testing with no recipe results...")
    no_results_response = llm.generate_recipe_response(
        user_query="Show me a recipe for flying spaghetti",
        recipe_results=[],
        return_json=True
    )
    print(f"\n✓ No Results Response:")
    print(json.dumps(no_results_response, indent=2))

    # Test 4: Test conversational response
    print("\n6. Testing conversational response...")
    conversation_history = [
        {"role": "user", "content": "How do I make carbonara?"},
        {"role": "assistant", "content": "Let me help you with Spaghetti Carbonara..."}
    ]
    context = {
        "recipe_title": "Simple Spaghetti Carbonara",
        "step_index": 2,
        "current_section": "steps",
        "paused": False
    }

    conversational_response = llm.generate_conversational_response(
        user_input="What's the next step?",
        intent="nav_next",
        conversation_history=conversation_history,
        context=context
    )
    print(f"\n✓ Conversational Response:")
    print(conversational_response)

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_structured_response()
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

