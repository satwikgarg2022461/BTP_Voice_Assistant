import os
import whisper
import glob
from datetime import datetime
import time


class WhisperASR:
    def __init__(self, model_size="base"):
        """
        Initialize Whisper ASR with the specified model size.
        
        Args:
            model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        print(f"Loading Whisper {model_size} model...")
        start_time = time.time()
        self.model = whisper.load_model(model_size)
        load_time = time.time() - start_time
        print(f"Whisper {model_size} model loaded in {load_time:.2f} seconds.")
    
    def transcribe_audio(self, audio_file_path):
        """
        Transcribe the given audio file.
        
        Args:
            audio_file_path (str): Path to the audio file to transcribe
            
        Returns:
            dict: Whisper transcription result containing 'text' and other metadata
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
        print(f"Transcribing: {audio_file_path}")
        start_time = time.time()
        result = self.model.transcribe(audio_file_path)
        transcription_time = time.time() - start_time
        print(f"Transcription completed in {transcription_time:.2f} seconds")
        
        return result
    
    def get_latest_recording(self, recordings_dir="voice_recordings"):
        """
        Get the path to the latest audio recording in the specified directory.
        
        Args:
            recordings_dir (str): Directory containing the recordings
            
        Returns:
            str: Path to the latest recording, or None if no recordings found
        """
        if not os.path.exists(recordings_dir):
            raise FileNotFoundError(f"Recordings directory not found: {recordings_dir}")
            
        recordings = glob.glob(os.path.join(recordings_dir, "*.wav"))
        
        if not recordings:
            return None
            
        latest_recording = max(recordings, key=os.path.getctime)
        return latest_recording
    
    def save_transcription(self, text, asr_text_dir="ASR_text", filename=None):
        """
        Save the transcription to a text file.
        
        Args:
            text (str): The transcribed text to save
            asr_text_dir (str): Directory to save the transcription
            filename (str, optional): Filename for the transcription, uses timestamp if None
            
        Returns:
            str: Path to the saved transcription file
        """
        os.makedirs(asr_text_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcription_{timestamp}.txt"
        
        file_path = os.path.join(asr_text_dir, filename)
        
        with open(file_path, "w") as f:
            f.write(text)
            
        return file_path