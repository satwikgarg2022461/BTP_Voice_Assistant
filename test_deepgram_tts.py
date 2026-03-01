#!/usr/bin/env python3
"""
Test script for Deepgram TTS integration
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.tts import RecipeTTS


def test_deepgram_tts():
    """Test Deepgram TTS basic functionality"""
    print("=" * 70)
    print("TESTING DEEPGRAM TTS INTEGRATION")
    print("=" * 70)

    # Initialize TTS
    print("\n1. Initializing Deepgram TTS...")
    tts = RecipeTTS()
    print("✓ TTS initialized with Deepgram")

    # Test 1: Simple speech generation
    print("\n2. Testing simple speech generation...")
    text = "Hello world! Today is a wonderful day to cook something delicious!"
    audio_path = tts.generate_speech(text, output_filename="test_deepgram_simple.mp3")

    if audio_path:
        print(f"✓ Audio generated: {audio_path}")
    else:
        print("✗ Failed to generate audio")

    # Test 2: Chunked generation with improved rate limiting
    print("\n3. Testing chunked generation...")
    recipe_text = """Welcome to making Spaghetti Carbonara! 
For ingredients, you'll need 400 grams of spaghetti.
You'll also need 4 eggs and parmesan cheese.
First, boil water and cook the spaghetti.
Meanwhile, beat the eggs in a bowl.
Cook the pancetta until crispy.
Finally, combine everything while hot and serve!"""

    chunks = tts.generate_speech_chunks(recipe_text, output_prefix="test_deepgram_chunked")

    print(f"\n✓ Generated {len(chunks)} chunks")
    successful = sum(1 for c in chunks if c.get("audio_path"))
    failed = sum(1 for c in chunks if c.get("error"))
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    # Test 3: Structured speech generation
    # print("\n4. Testing structured speech generation...")
    # structured_response = {
    #     "greeting": "Let's make a delicious pasta dish!",
    #     "ingredients": [
    #         {"text": "400 grams of spaghetti"},
    #         {"text": "4 large eggs"},
    #         {"text": "100 grams of parmesan cheese"}
    #     ],
    #     "steps": [
    #         {"step_num": 1, "text": "Boil water and cook spaghetti."},
    #         {"step_num": 2, "text": "Beat eggs and mix with cheese."},
    #         {"step_num": 3, "text": "Combine everything while hot."}
    #     ],
    #     "closing": "Enjoy your carbonara!"
    # }
    #
    # audio_metadata = tts.generate_structured_speech(structured_response, output_prefix="test_deepgram_structured")
    #
    # print(f"\n✓ Structured speech generated:")
    # print(f"  Greeting: {'✓' if audio_metadata['greeting'] else '✗'}")
    # print(f"  Ingredients: {len(audio_metadata['ingredients'])}")
    # print(f"  Steps: {len(audio_metadata['steps'])}")
    # print(f"  Closing: {'✓' if audio_metadata['closing'] else '✗'}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED!")
    print("=" * 70)


def main():
    """Run tests"""
    print("\n" + "=" * 70)
    print("DEEPGRAM TTS TEST SUITE")
    print("Testing Deepgram TTS API with improved rate limits (60 RPM)")
    print("=" * 70 + "\n")

    try:
        test_deepgram_tts()

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

