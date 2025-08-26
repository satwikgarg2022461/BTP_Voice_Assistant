# wakeword.py
import pvporcupine
import pyaudio
import struct
import os
import numpy as np
from scipy import signal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class WakeWordDetector:
    def __init__(self, access_key, keyword=None, model_path=None):
        # Create Porcupine instance with built-in keyword or custom file
        if model_path:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[model_path]
            )
        else:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[keyword] if keyword else ["porcupine"]
            )
        
        self.pa = pyaudio.PyAudio()
        
        # Get device information
        self.device_index = self._find_input_device()
        self.device_info = self.pa.get_device_info_by_index(self.device_index)
        self.device_sample_rate = int(self.device_info['defaultSampleRate'])
        print(f"Using input device: {self.device_index} with sample rate {self.device_sample_rate}")
        print(f"Porcupine requires sample rate: {self.porcupine.sample_rate}")
        
        # Calculate frame length for device sample rate
        self.device_frame_length = int((self.device_sample_rate / self.porcupine.sample_rate) * self.porcupine.frame_length)
        
        # Open the audio stream with the device's native sample rate
        self.stream = self.pa.open(
            rate=self.device_sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.device_frame_length,
            input_device_index=self.device_index
        )

    def _find_input_device(self):
        """Find a suitable input device that supports the required sample rate."""
        # Default to the default input device
        default_device_index = self.pa.get_default_input_device_info()['index']
        
        # Print available devices for debugging
        print("Available audio input devices:")
        for i in range(self.pa.get_device_count()):
            dev_info = self.pa.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:  # Only input devices
                print(f"Device {i}: {dev_info['name']}")
                print(f"  Max Input Channels: {dev_info['maxInputChannels']}")
                print(f"  Default Sample Rate: {dev_info['defaultSampleRate']}")
        
        # Return the default device index for now
        return default_device_index

    def _resample(self, audio_data, src_sample_rate, target_sample_rate):
        """Resample audio data from source to target sample rate."""
        # Convert bytes to numpy array
        audio_array = np.array(audio_data)
        
        # Calculate the resampling ratio
        ratio = target_sample_rate / src_sample_rate
        
        # Calculate the number of samples in the resampled audio
        new_length = int(len(audio_array) * ratio)
        
        # Resample the audio
        resampled = signal.resample(audio_array, new_length)
        
        return resampled.astype(np.int16)

    def listen(self):
        print("Listening for wake word...")
        while True:
            try:
                # Read audio from the stream at device's sample rate
                pcm_bytes = self.stream.read(self.device_frame_length, exception_on_overflow=False)
                
                # Convert bytes to PCM values
                pcm_device = struct.unpack_from("h" * self.device_frame_length, pcm_bytes)
                
                # Resample to Porcupine's required sample rate
                pcm_porcupine = self._resample(pcm_device, self.device_sample_rate, self.porcupine.sample_rate)
                
                # Process with Porcupine
                keyword_index = self.porcupine.process(pcm_porcupine)
                
                if keyword_index >= 0:
                    print("Wake word detected!")
                    return True
            except Exception as e:
                print(f"Error processing audio: {e}")
                continue
            
    def close(self):
        if hasattr(self, 'stream') and self.stream:
            self.stream.close()
        if hasattr(self, 'pa') and self.pa:
            self.pa.terminate()
        if hasattr(self, 'porcupine') and self.porcupine:
            self.porcupine.delete()

if __name__ == "__main__":
    # Load access key from environment variables
    access_key = os.getenv("PORCUPINE_ACCESS_KEY")
    if not access_key:
        raise ValueError("PORCUPINE_ACCESS_KEY not found in .env file")
    
    # Path to custom model file
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             "models", "Hey-Cook_en_linux_v3_0_0.ppn")
    
    try:
        detector = WakeWordDetector(access_key=access_key, model_path=model_path)
        detector.listen()
    finally:
        if 'detector' in locals():
            detector.close()