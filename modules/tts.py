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
        Generate speech and play it using the controllable AudioPlayer.

        Replaces the old subprocess.Popen approach with pygame.mixer so that
        playback can be paused, resumed, or stopped at any time.

        Args:
            text (str): The text to convert to speech.
            output_filename (str, optional): Custom filename for the output audio.

        Returns:
            tuple[str | None, AudioPlayer | None]:
                (path_to_audio_file, audio_player_instance)
                The caller keeps a reference to the AudioPlayer to pause/stop later.
                Returns (None, None) on error.
        """
        from modules.audio_player import AudioPlayer  # lazy import to avoid circular deps

        audio_path = self.generate_speech(text, output_filename)

        if not audio_path:
            return None, None

        try:
            player = AudioPlayer()
            player.play(audio_path)
            print(f"▶  Playing audio via AudioPlayer: {audio_path}")
            return audio_path, player
        except Exception as e:
            print(f"Could not play audio: {str(e)}")
            print(f"Audio file saved at: {audio_path}")
            return audio_path, None

    def split_into_sentences(self, text: str, max_length: int = 600) -> List[str]:
        """
        Split text into sentences for chunked TTS processing.

        Strategy
        --------
        1. Split ONLY on genuine sentence-ending punctuation (. ! ?)
           followed by whitespace + capital letter, or at end of string.
           Commas, semicolons, and the word "and" are intentionally NOT
           used as split points — they occur inside ingredient lists and
           step descriptions and must stay together for natural TTS output.

        2. For sentences that still exceed max_length after step 1,
           split on "; " boundaries, then merge comma-separated fragments
           up to max_length.

        Args:
            text (str): Full text to convert to speech.
            max_length (int): Max chars per chunk (default 600).

        Returns:
            List[str]: Sentence chunks.
        """
        if not text or not text.strip():
            return []

        sentence_endings = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])'
            r'|(?<=[.!?])\s*$'
            r'|\n+'
        )

        raw_sentences = sentence_endings.split(text.strip())

        sentences = []
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) <= max_length:
                sentences.append(sentence)
            else:
                sub_chunks = re.split(r';\s+', sentence)
                for chunk in sub_chunks:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    if len(chunk) <= max_length:
                        sentences.append(chunk)
                    else:
                        parts = [p.strip() for p in re.split(r',\s+', chunk) if p.strip()]
                        buf = parts[0] if parts else ""
                        for part in parts[1:]:
                            candidate = buf + ", " + part
                            if len(candidate) <= max_length:
                                buf = candidate
                            else:
                                sentences.append(buf[:max_length])
                                buf = part
                        if buf:
                            sentences.append(buf[:max_length])

        return sentences

    def generate_speech_chunks(self, text: str, output_prefix: Optional[str] = None,
                               max_workers: int = 6) -> List[Dict]:
        """
        Generate speech for text split into sentence chunks.

        All Deepgram API calls are fired **in parallel** via a thread pool so the
        total latency is roughly that of the single slowest chunk instead of the
        sum of all chunks.

        Args:
            text (str): Full text to convert to speech
            output_prefix (str, optional): Prefix for output filenames
            max_workers (int): Max parallel Deepgram requests (default 6)

        Returns:
            List[Dict]: Ordered list of chunk metadata dicts.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        sentences = self.split_into_sentences(text)
        if not sentences:
            print("No sentences to generate speech for")
            return []

        print(f"Split text into {len(sentences)} sentence chunk(s) — generating in parallel…")

        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"recipe_chunk_{timestamp}"

        def _generate_one(i: int, sentence: str) -> Dict:
            chunk_filename = f"{output_prefix}_chunk{i:03d}.mp3"
            audio_path = self.generate_speech(
                sentence, output_filename=chunk_filename, audio_format="mp3"
            )
            words_estimate = len(sentence) / 5
            duration_estimate = round((words_estimate / 150) * 60, 2)
            if audio_path:
                print(f"✓ Chunk {i+1}/{len(sentences)} ready")
                return {
                    "chunk_index": i,
                    "text": sentence,
                    "audio_path": audio_path,
                    "played": False,
                    "duration_estimate": duration_estimate,
                    "generated_at": datetime.now().isoformat(),
                }
            else:
                print(f"✗ Chunk {i+1}/{len(sentences)} failed")
                return {
                    "chunk_index": i,
                    "text": sentence,
                    "audio_path": None,
                    "played": False,
                    "duration_estimate": 0,
                    "generated_at": datetime.now().isoformat(),
                    "error": True,
                }

        # Fire all requests in parallel; re-sort by original sentence order
        results: List[Dict] = [{}] * len(sentences)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(sentences))) as pool:
            futures = {pool.submit(_generate_one, i, s): i for i, s in enumerate(sentences)}
            for future in as_completed(futures):
                meta = future.result()
                results[meta["chunk_index"]] = meta

        print(f"\n✓ Generated {len(results)} audio chunk(s) in parallel")
        return results

    def generate_speech_streaming(self, text: str, output_prefix: Optional[str] = None) -> Generator[Dict, None, None]:
        """
        Yield chunk 0 immediately for low-latency first audio, then generate
        the remaining chunks in parallel and yield them in order.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        sentences = self.split_into_sentences(text)

        if not sentences:
            print("No sentences to generate speech for")
            return

        total = len(sentences)
        print(f"Streaming TTS for {total} sentence chunks")

        if output_prefix is None:
            output_prefix = f"recipe_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        def _make_meta(i, sentence, audio_path):
            dur = round((len(sentence) / 5 / 150) * 60, 2)
            if audio_path:
                return {"chunk_index": i, "text": sentence, "audio_path": audio_path,
                        "played": False, "duration_estimate": dur,
                        "generated_at": datetime.now().isoformat(), "total_chunks": total}
            return {"chunk_index": i, "text": sentence, "audio_path": None,
                    "played": False, "duration_estimate": 0,
                    "generated_at": datetime.now().isoformat(), "error": True, "total_chunks": total}

        # --- Chunk 0: generate synchronously for instant playback ---
        print(f"\nStreaming chunk 1/{total}: {sentences[0][:60]}...")
        path0 = self.generate_speech(sentences[0], output_filename=f"{output_prefix}_chunk000.mp3", audio_format="mp3")
        yield _make_meta(0, sentences[0], path0)

        if total == 1:
            return

        # --- Chunks 1..N: fire in parallel, yield in order ---
        results: dict = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {}
            for i in range(1, total):
                fn = f"{output_prefix}_chunk{i:03d}.mp3"
                futures[pool.submit(self.generate_speech, sentences[i], fn, "mp3")] = i

            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"✗ Chunk {idx+1} error: {e}")
                    results[idx] = None

        for i in range(1, total):
            yield _make_meta(i, sentences[i], results.get(i))

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


# ─────────────────────────────────────────────────────────────────────────────
# Sarvam TTS — drop-in replacement for RecipeTTS
# ─────────────────────────────────────────────────────────────────────────────

class SarvamTTS:
    """
    Text-to-Speech using the Sarvam AI API (bulbul:v3 model).

    Public interface mirrors RecipeTTS so that callers in main.py
    can swap providers without any other changes.

    Sarvam limits: ~500 characters per request.
    Output format : WAV (saved as .wav files).
    """

    # Sarvam hard limit per API call
    SARVAM_CHAR_LIMIT = 500

    def __init__(self, output_dir: str = "tts_generated_speech",
                 language_code: str = "en-IN",
                 speaker: str = "ritu",
                 model: str = "bulbul:v3"):
        load_dotenv()

        self.api_key = os.getenv("Sarvam_API_key")
        if not self.api_key:
            raise ValueError(
                "Sarvam API key not found. Please set Sarvam_API_key in .env file."
            )

        from sarvamai import SarvamAI
        self._sarvam_client = SarvamAI(api_subscription_key=self.api_key)

        self.language_code = language_code
        self.speaker = speaker
        self.model = model

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Sarvam TTS initialized — model={model}, speaker={speaker}, "
              f"lang={language_code}")
        print(f"Audio files will be saved to: {self.output_dir}")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _parse_retry_delay(self, error_message: str) -> float:
        match = re.search(r'retry in ([\d.]+)s', str(error_message))
        if match:
            return float(match.group(1))
        match = re.search(r'retry after ([\d.]+)', str(error_message), re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 2.0

    # ── core TTS call ─────────────────────────────────────────────────────────

    def generate_speech(self, text: str, output_filename: Optional[str] = None,
                        audio_format: str = "wav", max_retries: int = 3) -> Optional[str]:
        """
        Generate speech from text using Sarvam AI TTS.

        Args:
            text (str): Text to convert (≤ 500 chars recommended).
            output_filename (str, optional): Target filename. Uses timestamp if None.
            audio_format (str): Ignored — Sarvam always returns WAV. Kept for
                                interface compatibility.
            max_retries (int): Retries on rate-limit errors.

        Returns:
            str: Path to saved .wav file, or None on error.
        """
        if not text or not text.strip():
            print("Error: Empty text provided for TTS")
            return None

        # Truncate silently if somehow still over limit
        text = text[:self.SARVAM_CHAR_LIMIT]

        print(f"Generating speech (Sarvam) for: {text[:80]}…")

        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_filename = f"sarvam_response_{timestamp}.wav"

        # Normalise extension — always .wav
        if not output_filename.lower().endswith(".wav"):
            output_filename = output_filename.rsplit(".", 1)[0] + ".wav"

        output_path = Path(self.output_dir) / output_filename

        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            try:
                from sarvamai.play import save as sarvam_save

                audio = self._sarvam_client.text_to_speech.convert(
                    target_language_code=self.language_code,
                    text=text,
                    model=self.model,
                    speaker=self.speaker,
                )
                sarvam_save(audio, str(output_path))

                print(f"✓ Speech generated (Sarvam) → {output_path}")
                return str(output_path)

            except Exception as e:
                error_str = str(e)
                last_error = e
                print(e)

                if "429" in error_str or "rate" in error_str.lower() or "quota" in error_str.lower():
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"✗ Max retries ({max_retries}) reached.")
                        return None
                    delay = self._parse_retry_delay(error_str) * (2 ** (retry_count - 1))
                    print(f"⚠️  Rate limit. Retry {retry_count}/{max_retries} in {delay:.1f}s…")
                    time.sleep(delay)
                else:
                    print(f"✗ Sarvam TTS error: {error_str}")
                    return None

        print(f"✗ Failed after {max_retries} retries: {last_error}")
        return None

    def generate_and_play_speech(self, text: str,
                                 output_filename: Optional[str] = None):
        """Generate speech and play it via AudioPlayer."""
        from modules.audio_player import AudioPlayer

        audio_path = self.generate_speech(text, output_filename)
        if not audio_path:
            return None, None

        try:
            player = AudioPlayer()
            player.play(audio_path)
            print(f"▶ Playing audio: {audio_path}")
            return audio_path, player
        except Exception as e:
            print(f"Could not play audio: {e}")
            return audio_path, None

    # ── sentence splitting ────────────────────────────────────────────────────

    def split_into_sentences(self, text: str, max_length: int = 0) -> List[str]:
        """
        Same strategy as RecipeTTS.split_into_sentences but defaults
        max_length to SARVAM_CHAR_LIMIT to respect the API constraint.
        """
        if max_length <= 0:
            max_length = self.SARVAM_CHAR_LIMIT

        if not text or not text.strip():
            return []

        sentence_endings = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])'
            r'|(?<=[.!?])\s*$'
            r'|\n+'
        )

        raw_sentences = sentence_endings.split(text.strip())

        sentences: List[str] = []
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) <= max_length:
                sentences.append(sentence)
            else:
                sub_chunks = re.split(r';\s+', sentence)
                for chunk in sub_chunks:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    if len(chunk) <= max_length:
                        sentences.append(chunk)
                    else:
                        parts = [p.strip() for p in re.split(r',\s+', chunk) if p.strip()]
                        buf = parts[0] if parts else ""
                        for part in parts[1:]:
                            candidate = buf + ", " + part
                            if len(candidate) <= max_length:
                                buf = candidate
                            else:
                                sentences.append(buf[:max_length])
                                buf = part
                        if buf:
                            sentences.append(buf[:max_length])

        return sentences

    # ── chunked (parallel) generation ────────────────────────────────────────

    def generate_speech_chunks(self, text: str, output_prefix: Optional[str] = None,
                               max_workers: int = 6) -> List[Dict]:
        """
        Generate speech for all sentence chunks in parallel.
        Interface identical to RecipeTTS.generate_speech_chunks.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        sentences = self.split_into_sentences(text)
        if not sentences:
            print("No sentences to generate speech for")
            return []

        print(f"Split text into {len(sentences)} chunk(s) — generating in parallel (Sarvam)…")

        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"sarvam_chunk_{timestamp}"

        def _generate_one(i: int, sentence: str) -> Dict:
            chunk_filename = f"{output_prefix}_chunk{i:03d}.wav"
            audio_path = self.generate_speech(sentence, output_filename=chunk_filename)
            words_estimate = len(sentence) / 5
            duration_estimate = round((words_estimate / 150) * 60, 2)
            if audio_path:
                print(f"✓ Chunk {i+1}/{len(sentences)} ready")
                return {
                    "chunk_index": i,
                    "text": sentence,
                    "audio_path": audio_path,
                    "played": False,
                    "duration_estimate": duration_estimate,
                    "generated_at": datetime.now().isoformat(),
                }
            print(f"✗ Chunk {i+1}/{len(sentences)} failed")
            return {
                "chunk_index": i,
                "text": sentence,
                "audio_path": None,
                "played": False,
                "duration_estimate": 0,
                "generated_at": datetime.now().isoformat(),
                "error": True,
            }

        results: List[Dict] = [{}] * len(sentences)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(sentences))) as pool:
            futures = {pool.submit(_generate_one, i, s): i for i, s in enumerate(sentences)}
            for future in futures:
                meta = future.result()
                results[meta["chunk_index"]] = meta

        print(f"\n✓ Generated {len(results)} audio chunk(s) in parallel (Sarvam)")
        return results

    # ── streaming generation ──────────────────────────────────────────────────

    def generate_speech_streaming(self, text: str,
                                  output_prefix: Optional[str] = None) -> Generator[Dict, None, None]:
        """
        Yield chunk 0 immediately for low-latency first audio, then generate
        the remaining chunks in parallel and yield them in order.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        sentences = self.split_into_sentences(text)
        if not sentences:
            print("No sentences to generate speech for")
            return

        total = len(sentences)
        print(f"Streaming TTS (Sarvam) for {total} sentence chunks")

        if output_prefix is None:
            output_prefix = f"sarvam_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        def _make_meta(i, sentence, audio_path):
            dur = round((len(sentence) / 5 / 150) * 60, 2)
            if audio_path:
                return {"chunk_index": i, "text": sentence, "audio_path": audio_path,
                        "played": False, "duration_estimate": dur,
                        "generated_at": datetime.now().isoformat(), "total_chunks": total}
            return {"chunk_index": i, "text": sentence, "audio_path": None,
                    "played": False, "duration_estimate": 0,
                    "generated_at": datetime.now().isoformat(), "error": True, "total_chunks": total}

        # --- Chunk 0: generate synchronously for instant playback ---
        print(f"\nStreaming chunk 1/{total}: {sentences[0][:60]}…")
        path0 = self.generate_speech(sentences[0], output_filename=f"{output_prefix}_chunk000.wav")
        yield _make_meta(0, sentences[0], path0)

        if total == 1:
            return

        # --- Chunks 1..N: fire in parallel, yield in order ---
        results: dict = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {}
            for i in range(1, total):
                fn = f"{output_prefix}_chunk{i:03d}.wav"
                futures[pool.submit(self.generate_speech, sentences[i], fn)] = i

            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"✗ Chunk {idx+1} error: {e}")
                    results[idx] = None

        for i in range(1, total):
            yield _make_meta(i, sentences[i], results.get(i))

    # ── playback tracking helpers (same as RecipeTTS) ─────────────────────────

    def mark_chunk_played(self, chunks_metadata: List[Dict], chunk_index: int) -> bool:
        if 0 <= chunk_index < len(chunks_metadata):
            chunks_metadata[chunk_index]["played"] = True
            return True
        return False

    def get_unplayed_chunks(self, chunks_metadata: List[Dict]) -> List[Dict]:
        return [c for c in chunks_metadata if not c.get("played", False)]

    def get_playback_progress(self, chunks_metadata: List[Dict]) -> Dict:
        total = len(chunks_metadata)
        played = sum(1 for c in chunks_metadata if c.get("played", False))
        remaining = total - played
        percentage = (played / total * 100) if total > 0 else 0
        return {
            "total_chunks": total,
            "played_chunks": played,
            "remaining_chunks": remaining,
            "progress_percentage": round(percentage, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_tts_engine(provider: str = "deepgram", **kwargs):
    """
    Return a TTS engine instance for the requested provider.

    Args:
        provider (str): "deepgram" (default) or "sarvam".
        **kwargs: Forwarded to the engine constructor.

    Returns:
        RecipeTTS | SarvamTTS
    """
    provider = provider.lower().strip()
    if provider == "sarvam":
        return SarvamTTS(**kwargs)
    elif provider == "deepgram":
        return RecipeTTS(**kwargs)
    else:
        raise ValueError(f"Unknown TTS provider '{provider}'. Choose 'deepgram' or 'sarvam'.")
