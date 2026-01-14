import os
import time
import threading
from dotenv import load_dotenv

from deepgram import DeepgramClient
from deepgram.core.events import EventType

# Load environment variables
load_dotenv()


class DeepgramASR:
    def __init__(self, api_key=None):
        """
        Initialize Deepgram ASR with API key.
        """
        print("Initializing Deepgram ASR (Streaming)...")

        self.api_key = api_key or os.getenv("Deepgram_API_key")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key not found. Please set Deepgram_API_key in .env file or pass it as parameter."
            )

        self.client = DeepgramClient(api_key=self.api_key)
        print("Deepgram ASR initialized successfully!")

    def transcribe_audio(self, audio_file_path):
        """
        Transcribe an audio file using Deepgram Streaming API.

        Args:
            audio_file_path (str): Path to the audio file

        Returns:
            dict: Whisper-compatible transcription output
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        print(f"Streaming transcription started: {audio_file_path}")
        start_time = time.time()

        transcripts = []
        ready = threading.Event()

        def on_message(result):
            channel = getattr(result, "channel", None)
            if channel and hasattr(channel, "alternatives"):
                alt = channel.alternatives[0]
                transcript = alt.transcript
                is_final = getattr(result, "is_final", True)

                if transcript and is_final:
                    transcripts.append(transcript)
                    print(transcript)

        with self.client.listen.v1.connect(
            model="nova-3",
            language="en",
            smart_format=True,
            punctuate=True,
        ) as connection:

            connection.on(EventType.OPEN, lambda _: ready.set())
            connection.on(EventType.MESSAGE, on_message)

            def stream_audio():
                ready.wait()
                with open(audio_file_path, "rb") as audio:
                    while True:
                        chunk = audio.read(4096)
                        if not chunk:
                            break
                        connection.send_media(chunk)


            threading.Thread(target=stream_audio, daemon=True).start()
            connection.start_listening()

        transcription_time = time.time() - start_time
        final_text = " ".join(transcripts)

        print(f"Transcription completed in {transcription_time:.2f} seconds")

        return {
            "text": final_text,
            "language": "en",
            "duration": transcription_time,
        }

    def get_latest_recording(self, recordings_dir="voice_recordings"):
        """
        Get the latest audio recording from a directory.
        """
        import glob

        if not os.path.exists(recordings_dir):
            raise FileNotFoundError(f"Recordings directory not found: {recordings_dir}")

        recordings = glob.glob(os.path.join(recordings_dir, "*.wav"))
        if not recordings:
            return None

        return max(recordings, key=os.path.getctime)

    def save_transcription(self, text, asr_text_dir="ASR_text", filename=None):
        """
        Save transcription to a text file.
        """
        from datetime import datetime

        os.makedirs(asr_text_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcription_{timestamp}.txt"

        file_path = os.path.join(asr_text_dir, filename)

        with open(file_path, "w") as f:
            f.write(text)

        return file_path
