import os
from datetime import datetime
from dotenv import load_dotenv

import pvporcupine
from pvrecorder import PvRecorder


class WakeWordDetector:
    def __init__(self, keyword_paths, sensitivity=0.65, device_index=-1):
        """
        Wake Word Detector using Picovoice Porcupine + PvRecorder.

        Args:
            keyword_paths (list[str]): List of custom .ppn wake word model paths
            sensitivity (float or list[float]): Detection sensitivity (0.0–1.0)
            device_index (int): Audio device index (-1 = default)
        """
        # Load Access Key
        load_dotenv()
        access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        if not access_key:
            raise ValueError("Missing PORCUPINE_ACCESS_KEY in .env file")

        if isinstance(sensitivity, float):
            sensitivity = [sensitivity] * len(keyword_paths)

        # Initialize Porcupine
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=keyword_paths,
            sensitivities=sensitivity
        )

        # Initialize recorder
        self.recorder = PvRecorder(
            frame_length=self.porcupine.frame_length,
            device_index=device_index
        )

        print(f"Porcupine version: {self.porcupine.version}")

    def start(self, on_detect=None):
        """
        Start listening for wake words.

        Args:
            on_detect (callable): Callback when wake word detected.
                                  Signature: on_detect(keyword_index: int, timestamp: datetime)
        """
        print("Listening for wake word... Press Ctrl+C to stop.")
        self.recorder.start()

        try:
            while True:
                pcm = self.recorder.read()
                result = self.porcupine.process(pcm)

                if result >= 0:
                    timestamp = datetime.now()
                    if on_detect:
                        on_detect(result, timestamp)
                    else:
                        print(f"[{timestamp}] Wake word detected! (index={result})")
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

    def stop(self):
        """Clean up resources."""
        self.recorder.stop()
        self.recorder.delete()
        self.porcupine.delete()


# Example usage
if __name__ == "__main__":
    detector = WakeWordDetector(
        keyword_paths=["models/Hey-Cook_en_linux_v3_0_0.ppn"],
        sensitivity=0.65,
        device_index=-1
    )

    def on_wake(keyword_index, timestamp):
        print(f"[{timestamp}] Hey Cook detected! (keyword index: {keyword_index})")

    detector.start(on_detect=on_wake)
