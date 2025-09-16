import os
import wave
import numpy as np
from collections import deque
from datetime import datetime
from pvrecorder import PvRecorder


class ShortRecorder:
    def __init__(self, sample_rate=16000, frame_length=512, pre_roll_secs=1.0,
                 silence_thresh=500, silence_duration=3.0):
        """
        Args:
            sample_rate (int): Target sample rate (16 kHz for ASR).
            frame_length (int): Frame size (matches Porcupine/PvRecorder).
            pre_roll_secs (float): Seconds of audio to keep before trigger.
            silence_thresh (int): RMS threshold for silence.
            silence_duration (float): How long silence lasts before stopping (seconds).
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.pre_roll_frames = int((pre_roll_secs * sample_rate) // frame_length)
        self.silence_thresh = silence_thresh
        self.silence_frames = int((silence_duration * sample_rate) // frame_length)

        self.recorder = PvRecorder(frame_length=frame_length, device_index=-1)

    def rms(self, pcm):
        """Compute root mean square of frame."""
        arr = np.array(pcm, dtype=np.int16)
        return np.sqrt(np.mean(arr.astype(np.float32) ** 2))

    def record_once(self, out_path="temp.wav"):
        """
        Record one utterance with pre-roll and silence detection.
        Returns: saved .wav file path
        """
        print("Recording... speak now!")

        # Buffers
        pre_buffer = deque(maxlen=self.pre_roll_frames)
        audio_frames = []

        # Start mic
        self.recorder.start()

        # Pre-roll fill
        for _ in range(self.pre_roll_frames):
            pre_buffer.append(self.recorder.read())

        silence_counter = 0
        started = True  # assume wake word already triggered

        while True:
            pcm = self.recorder.read()

            if started:
                audio_frames.append(pcm)

            # Check silence
            energy = self.rms(pcm)
            if energy < self.silence_thresh:
                silence_counter += 1
            else:
                silence_counter = 0

            # Stop after silence
            if silence_counter >= self.silence_frames:
                break

        self.recorder.stop()

        # Combine pre-roll + recorded audio
        full_audio = list(pre_buffer) + audio_frames

        # Save WAV
        flat_pcm = np.concatenate([np.array(f, dtype=np.int16) for f in full_audio])
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(flat_pcm.tobytes())

        print(f"Saved recording to {out_path}")
        return out_path


# Example usage
if __name__ == "__main__":
    recorder = ShortRecorder(sample_rate=16000, frame_length=512,
                             pre_roll_secs=1.0, silence_thresh=500,
                             silence_duration=3.0)
    file_path = recorder.record_once("voice_recordings/temp.wav")
    print("Pass this file to ASR:", file_path)
