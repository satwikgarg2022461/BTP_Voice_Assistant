#!/usr/bin/env python3
import os
import sys
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.wakeword import WakeWordDetector
from modules.vad import ShortRecorder
from modules.asr import WhisperASR


class VoiceAssistant:
    def __init__(self, 
                 keyword_paths=["models/Hey-Cook_en_linux_v3_0_0.ppn"],
                 wake_sensitivity=0.65,
                 device_index=-1,
                 recordings_dir="voice_recordings",
                 asr_model="base",
                 asr_text_dir="ASR_text"):
        """
        Initialize the voice assistant that integrates wake word detection and VAD recording.
        
        Args:
            keyword_paths (list): Paths to wake word model files
            wake_sensitivity (float): Wake word detection sensitivity
            device_index (int): Audio device index (-1 for default)
            recordings_dir (str): Directory to save voice recordings
            asr_model (str): Whisper model size to use
            asr_text_dir (str): Directory to save transcription outputs
        """
        print("Initializing Voice Assistant...")
        
        # Create recordings directory if it doesn't exist
        self.recordings_dir = recordings_dir
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Create ASR text directory if it doesn't exist
        self.asr_text_dir = asr_text_dir
        os.makedirs(self.asr_text_dir, exist_ok=True)
        
        # Initialize wake word detector
        self.wake_detector = WakeWordDetector(
            keyword_paths=keyword_paths,
            sensitivity=wake_sensitivity,
            device_index=device_index
        )
        
        # Initialize VAD recorder
        self.recorder = ShortRecorder(
            sample_rate=16000,
            frame_length=512,
            pre_roll_secs=1.0,
            silence_thresh=900,
            silence_duration=1.0
        )
        
        # Initialize ASR
        self.asr = WhisperASR(model_size=asr_model)
        
        print("Voice Assistant ready!")

    def on_wake_word(self, keyword_index, timestamp):
        """Callback function when wake word is detected"""
        print(f"\n[{timestamp}] Wake word detected! Listening...")
        
        # Generate a filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_path = os.path.join(self.recordings_dir, f"recording_{timestamp_str}.wav")
        
        # Start VAD recording
        audio_path = self.recorder.record_once(recording_path)
        
        print(f"Recording saved to: {audio_path}")
        
        # Transcribe the audio
        try:
            print("Transcribing audio...")
            result = self.asr.transcribe_audio(audio_path)
            transcribed_text = result["text"]
            
            # Display the transcription
            print("\n--- Transcription Result ---")
            print(transcribed_text)
            print("---------------------------\n")
            
            # Save transcription
            audio_filename = os.path.basename(audio_path)
            output_filename = f"{os.path.splitext(audio_filename)[0]}.txt"
            output_path = self.asr.save_transcription(
                transcribed_text,
                asr_text_dir=self.asr_text_dir,
                filename=output_filename
            )
            print(f"Transcription saved to: {output_path}")
            
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
        
    def run(self):
        """Run the voice assistant in continuous mode"""
        print("Starting voice assistant. Say the wake word to begin recording.")
        print("Press Ctrl+C to exit.")
        
        try:
            self.wake_detector.start(on_detect=self.on_wake_word)
        except KeyboardInterrupt:
            print("\nShutting down voice assistant...")
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            # Cleanup will be handled by wake_detector.stop()
            pass


if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()