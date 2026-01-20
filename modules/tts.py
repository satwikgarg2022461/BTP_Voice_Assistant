import os
import re
import time
from datetime import datetime
from pathlib import Path
from deepgram import DeepgramClient
from dotenv import load_dotenv
from typing import List, Dict, Generator, Optional, Tuple

class RecipeTTS:
    def __init__(self, output_dir="tts_generated_speech"):
        """
        Initialize the Recipe Text-to-Speech module using Deepgram TTS API

        Args:
            output_dir (str): Directory to save generated speech files
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize Deepgram client (API key from environment variables)
        self.api_key = os.getenv("Deepgram_API_key")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key not found. Please set Deepgram_API_key in .env file or pass it as parameter."
            )

        self.client = DeepgramClient(api_key=self.api_key)
        print("Deepgram TTS initialized successfully!")
        self.model_name = "aura-2-thalia-en"  # Deepgram Aura model

        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"TTS module initialized. Audio files will be saved to: {self.output_dir}")
        print(f"Using Deepgram TTS model: {self.model_name}")


    def _parse_retry_delay(self, error_message: str) -> float:
        """
        Parse retry delay from API error message

        Args:
            error_message (str): Error message from API

        Returns:
            float: Retry delay in seconds, or default value
        """
        import re

        # Try to extract retry delay from error message
        match = re.search(r'retry in ([\d.]+)s', str(error_message))
        if match:
            return float(match.group(1))

        # Try to extract from "Retry after X seconds"
        match = re.search(r'retry after ([\d.]+)', str(error_message), re.IGNORECASE)
        if match:
            return float(match.group(1))

        # Default to 2 seconds for Deepgram (faster than Gemini)
        return 2.0

    # def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    #     with wave.open(filename, "wb") as wf:
    #         wf.setnchannels(channels)
    #         wf.setsampwidth(sample_width)
    #         wf.setframerate(rate)
    #         wf.writeframes(pcm)

    def generate_speech(self, text, output_filename=None, audio_format="mp3", max_retries=3):
        """
        Generate speech from text using Deepgram TTS API with rate limiting and retry logic

        Args:
            text (str): The text to convert to speech
            output_filename (str, optional): Custom filename for the output audio. 
                                           If None, uses timestamp
            audio_format (str): Audio format ('mp3', 'wav', 'opus', 'flac', etc.).
                               Default is 'mp3' for Deepgram TTS.
            max_retries (int): Maximum number of retry attempts for rate limit errors

        Returns:
            str: Path to the saved audio file, or None on error
        """
        if not text or not text.strip():
            print("Error: Empty text provided for TTS")
            return None

        print(f"Generating speech for: {text[:100]}...")

        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = audio_format.lower()
            output_filename = f"recipe_response_{timestamp}.{ext}"

        # Ensure output filename has correct extension
        if not output_filename.endswith(f".{audio_format}"):
            output_filename = f"{output_filename.rsplit('.', 1)[0]}.{audio_format}"

        output_path = Path(self.output_dir) / output_filename

        # Retry loop with exponential backoff
        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            try:
                # Make API request using Deepgram official pattern
                response = self.client.speak.v1.audio.generate(
                    text=text,
                     model=self.model_name
                )

                # Save the audio file
                with open(output_path, "wb") as audio_file:
                    for a in response:
                        audio_file.write(a)

                print(f"✓ Speech generated and saved to: {output_path}")
                return str(output_path)

            except Exception as e:
                error_str = str(e)
                last_error = e
                print(e)

                # Check if it's a rate limit error (429)
                if "429" in error_str or "rate" in error_str.lower() or "quota" in error_str.lower():
                    retry_count += 1

                    if retry_count > max_retries:
                        print(f"✗ Max retries ({max_retries}) reached. Failed to generate speech.")
                        return None

                    # Parse retry delay from error message
                    retry_delay = self._parse_retry_delay(error_str)

                    # Add exponential backoff: base_delay * 2^(retry_count-1)
                    backoff_delay = retry_delay * (2 ** (retry_count - 1))

                    print(f"⚠️  Rate limit error. Retry {retry_count}/{max_retries} in {backoff_delay:.1f}s...")
                    time.sleep(backoff_delay)


                else:
                    # Non-rate-limit error, don't retry
                    print(f"✗ Error generating speech: {error_str}")
                    return None

        print(f"✗ Failed to generate speech after {max_retries} retries: {last_error}")
        return None

    def generate_and_play_speech(self, text, output_filename=None):
        """
        Generate speech and optionally play it using system audio player
        
        Args:
            text (str): The text to convert to speech
            output_filename (str, optional): Custom filename for the output audio
            
        Returns:
            str: Path to the saved audio file, or None on error
        """
        # First generate the speech
        audio_path = self.generate_speech(text, output_filename)
        
        if audio_path:
            # Try to play the audio using system player
            try:
                import subprocess
                # Use common audio players available on Linux
                players = ['mpv', 'ffplay', 'paplay', 'aplay']
                
                for player in players:
                    try:
                        subprocess.Popen([player, audio_path], 
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
                        print(f"Playing audio with {player}...")
                        return audio_path
                    except FileNotFoundError:
                        continue
                
                # If no player found, just return the path
                print(f"Audio file ready at: {audio_path}")
                print("(No audio player found. Please play the file manually.)")
                return audio_path
                
            except Exception as e:
                print(f"Could not play audio: {str(e)}")
                print(f"Audio file saved at: {audio_path}")
                return audio_path
        
        return None

    def split_into_sentences(self, text: str, max_length: int = 200) -> List[str]:
        """
        Split text into sentences for chunked TTS processing

        Args:
            text (str): Text to split
            max_length (int): Maximum length for a sentence chunk (will split long sentences)

        Returns:
            List[str]: List of sentence chunks
        """
        if not text or not text.strip():
            return []

        # First, split by common sentence boundaries
        # This regex handles: . ! ? followed by space or newline, but not decimals like 3.5
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$|\n+')

        # Split text
        raw_sentences = sentence_endings.split(text.strip())

        # Filter out empty strings and clean up
        sentences = []
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if sentence:
                # If sentence is too long, split it further
                if len(sentence) > max_length:
                    # Split by commas, semicolons, or conjunctions
                    sub_chunks = re.split(r'[,;]|\s+(?:and|or|but)\s+', sentence)
                    for chunk in sub_chunks:
                        chunk = chunk.strip()
                        if chunk:
                            sentences.append(chunk)
                else:
                    sentences.append(sentence)

        return sentences

    def generate_speech_chunks(self, text: str, output_prefix: Optional[str] = None) -> List[Dict]:
        """
        Generate speech for text split into sentence chunks (sequential, not parallel)

        Args:
            text (str): Full text to convert to speech
            output_prefix (str, optional): Prefix for output filenames

        Returns:
            List[Dict]: List of chunk metadata with format:
                {
                    "chunk_index": 0,
                    "text": "sentence text",
                    "audio_path": "path/to/audio.wav",
                    "played": False,
                    "duration_estimate": 3.5,  # seconds (rough estimate)
                    "generated_at": "timestamp"
                }
        """
        # Split text into sentences
        sentences = self.split_into_sentences(text)

        if not sentences:
            print("No sentences to generate speech for")
            return []

        print(f"Split text into {len(sentences)} sentence chunks")

        # Generate timestamp prefix if not provided
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"recipe_chunk_{timestamp}"

        chunks_metadata = []

        # Generate audio for each sentence sequentially
        for i, sentence in enumerate(sentences):
            print(f"\nGenerating chunk {i+1}/{len(sentences)}: {sentence[:60]}...")

            # Generate filename for this chunk (use mp3 for Deepgram)
            chunk_filename = f"{output_prefix}_chunk{i:03d}.mp3"

            # Generate speech for this chunk
            audio_path = self.generate_speech(sentence, output_filename=chunk_filename, audio_format="mp3")

            if audio_path:
                # Estimate duration (very rough: ~150 words per minute, ~5 chars per word)
                words_estimate = len(sentence) / 5
                duration_estimate = (words_estimate / 150) * 60  # seconds

                chunk_metadata = {
                    "chunk_index": i,
                    "text": sentence,
                    "audio_path": audio_path,
                    "played": False,
                    "duration_estimate": round(duration_estimate, 2),
                    "generated_at": datetime.now().isoformat()
                }

                chunks_metadata.append(chunk_metadata)
                print(f"✓ Chunk {i+1} generated: {audio_path}")
            else:
                print(f"✗ Failed to generate chunk {i+1}")
                # Add failed chunk to metadata for tracking
                chunks_metadata.append({
                    "chunk_index": i,
                    "text": sentence,
                    "audio_path": None,
                    "played": False,
                    "duration_estimate": 0,
                    "generated_at": datetime.now().isoformat(),
                    "error": True
                })

        print(f"\n✓ Generated {len(chunks_metadata)} audio chunks")
        return chunks_metadata

    def generate_speech_streaming(self, text: str, output_prefix: Optional[str] = None) -> Generator[Dict, None, None]:
        """
        Generate speech chunks as a generator, yielding each chunk as it's ready
        This allows for progressive playback (play first chunk while generating next)

        Args:
            text (str): Full text to convert to speech
            output_prefix (str, optional): Prefix for output filenames

        Yields:
            Dict: Chunk metadata for each generated audio chunk
        """
        # Split text into sentences
        sentences = self.split_into_sentences(text)

        if not sentences:
            print("No sentences to generate speech for")
            return

        print(f"Streaming TTS for {len(sentences)} sentence chunks")

        # Generate timestamp prefix if not provided
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"recipe_stream_{timestamp}"

        # Generate and yield each chunk sequentially
        for i, sentence in enumerate(sentences):
            print(f"\nStreaming chunk {i+1}/{len(sentences)}: {sentence[:60]}...")

            # Generate filename for this chunk (use mp3 for Deepgram)
            chunk_filename = f"{output_prefix}_chunk{i:03d}.mp3"

            # Generate speech for this chunk
            audio_path = self.generate_speech(sentence, output_filename=chunk_filename, audio_format="mp3")

            if audio_path:
                # Estimate duration
                words_estimate = len(sentence) / 5
                duration_estimate = (words_estimate / 150) * 60

                chunk_metadata = {
                    "chunk_index": i,
                    "text": sentence,
                    "audio_path": audio_path,
                    "played": False,
                    "duration_estimate": round(duration_estimate, 2),
                    "generated_at": datetime.now().isoformat(),
                    "total_chunks": len(sentences)
                }

                print(f"✓ Yielding chunk {i+1}")
                yield chunk_metadata
            else:
                print(f"✗ Failed to generate chunk {i+1}")
                yield {
                    "chunk_index": i,
                    "text": sentence,
                    "audio_path": None,
                    "played": False,
                    "duration_estimate": 0,
                    "generated_at": datetime.now().isoformat(),
                    "error": True,
                    "total_chunks": len(sentences)
                }

    def generate_structured_speech(self, structured_response: Dict, output_prefix: Optional[str] = None) -> Dict:
        """
        Generate speech for structured JSON response from LLM
        Handles greeting, ingredients, steps, and closing separately

        Args:
            structured_response (dict): Structured response with greeting, ingredients, steps, closing
            output_prefix (str, optional): Prefix for output filenames

        Returns:
            Dict: Structured metadata with audio paths and tracking info
                {
                    "greeting": {"text": "...", "audio_path": "...", "played": False},
                    "ingredients": [{"text": "...", "audio_path": "...", "played": False}, ...],
                    "steps": [{"step_num": 1, "text": "...", "audio_path": "...", "played": False}, ...],
                    "closing": {"text": "...", "audio_path": "...", "played": False}
                }
        """
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"recipe_structured_{timestamp}"

        result = {
            "greeting": None,
            "ingredients": [],
            "steps": [],
            "closing": None,
            "generated_at": datetime.now().isoformat()
        }

        # Generate greeting
        if structured_response.get("greeting"):
            print("\n=== Generating Greeting ===")
            greeting_text = structured_response["greeting"]
            audio_path = self.generate_speech(greeting_text, f"{output_prefix}_greeting.mp3", audio_format="mp3")
            result["greeting"] = {
                "text": greeting_text,
                "audio_path": audio_path,
                "played": False
            }

        # Generate ingredients
        ingredients = structured_response.get("ingredients", [])
        if ingredients:
            print(f"\n=== Generating {len(ingredients)} Ingredients ===")
            for i, ing in enumerate(ingredients):
                ing_text = ing.get("text", "")
                if ing_text:
                    audio_path = self.generate_speech(ing_text, f"{output_prefix}_ingredient_{i:03d}.mp3", audio_format="mp3")
                    result["ingredients"].append({
                        "index": i,
                        "text": ing_text,
                        "audio_path": audio_path,
                        "played": False
                    })

        # Generate steps
        steps = structured_response.get("steps", [])
        if steps:
            print(f"\n=== Generating {len(steps)} Steps ===")
            for step in steps:
                step_num = step.get("step_num", 0)
                step_text = step.get("text", "")
                if step_text:
                    audio_path = self.generate_speech(step_text, f"{output_prefix}_step_{step_num:03d}.mp3", audio_format="mp3")
                    result["steps"].append({
                        "step_num": step_num,
                        "text": step_text,
                        "audio_path": audio_path,
                        "played": False
                    })

        # Generate closing
        if structured_response.get("closing"):
            print("\n=== Generating Closing ===")
            closing_text = structured_response["closing"]
            audio_path = self.generate_speech(closing_text, f"{output_prefix}_closing.mp3", audio_format="mp3")
            result["closing"] = {
                "text": closing_text,
                "audio_path": audio_path,
                "played": False
            }

        print(f"\n✓ Structured speech generation complete")
        return result

    def mark_chunk_played(self, chunks_metadata: List[Dict], chunk_index: int) -> bool:
        """
        Mark a specific chunk as played

        Args:
            chunks_metadata (list): List of chunk metadata dicts
            chunk_index (int): Index of chunk to mark

        Returns:
            bool: Success status
        """
        if 0 <= chunk_index < len(chunks_metadata):
            chunks_metadata[chunk_index]["played"] = True
            return True
        return False

    def get_unplayed_chunks(self, chunks_metadata: List[Dict]) -> List[Dict]:
        """
        Get list of chunks that haven't been played yet

        Args:
            chunks_metadata (list): List of chunk metadata dicts

        Returns:
            List[Dict]: List of unplayed chunks
        """
        return [chunk for chunk in chunks_metadata if not chunk.get("played", False)]

    def get_playback_progress(self, chunks_metadata: List[Dict]) -> Dict:
        """
        Get playback progress statistics

        Args:
            chunks_metadata (list): List of chunk metadata dicts

        Returns:
            Dict: Progress info with total, played, remaining, percentage
        """
        total = len(chunks_metadata)
        played = sum(1 for chunk in chunks_metadata if chunk.get("played", False))
        remaining = total - played
        percentage = (played / total * 100) if total > 0 else 0

        return {
            "total_chunks": total,
            "played_chunks": played,
            "remaining_chunks": remaining,
            "progress_percentage": round(percentage, 1)
        }
