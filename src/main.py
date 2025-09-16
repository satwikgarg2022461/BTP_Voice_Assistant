#!/usr/bin/env python3
import os
import sys
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.wakeword import WakeWordDetector
from modules.vad import ShortRecorder


class VoiceAssistant:
    def __init__(self, 
                 keyword_paths=["models/Hey-Cook_en_linux_v3_0_0.ppn"],
                 wake_sensitivity=0.65,
                 device_index=-1,
                 recordings_dir="voice_recordings"):
        """
        Initialize the voice assistant that integrates wake word detection and VAD recording.
        
        Args:
            keyword_paths (list): Paths to wake word model files
            wake_sensitivity (float): Wake word detection sensitivity
            device_index (int): Audio device index (-1 for default)
            recordings_dir (str): Directory to save voice recordings
        """
        print("Initializing Voice Assistant...")
        
        # Create recordings directory if it doesn't exist
        self.recordings_dir = recordings_dir
        os.makedirs(self.recordings_dir, exist_ok=True)
        
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
            silence_thresh=500,
            silence_duration=3.0
        )
        
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
        
        # Here you could add processing for the recording
        # For example, send to ASR, process with LLM, etc.
        
    def run(self):
        """Run the voice assistant in continuous mode"""
        print("Starting voice assistant. Say the wake word to begin recording.")
        print("Press Ctrl+C to exit.")
        
        try:
            self.wake_detector.start(on_detect=self.on_wake_word)
        except KeyboardInterrupt:
            print("\nShutting down voice assistant...")
        finally:
            # Cleanup will be handled by wake_detector.stop()
            pass


if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()