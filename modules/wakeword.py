import os
import threading
from datetime import datetime
from dotenv import load_dotenv

import pvporcupine
from pvrecorder import PvRecorder


class WakeWordDetector:
    """
    Wake Word Detector using Picovoice Porcupine + PvRecorder.

    Runs the Porcupine listen-loop in a **background daemon thread** so it
    never blocks the main thread.  Uses threading primitives:

      detected_event  – set() when the wake word fires; cleared after the
                        main loop acknowledges it via acknowledge().
      _stop_event     – set() to ask the background thread to exit cleanly.

    Typical usage
    -------------
        detector = WakeWordDetector(["models/Hey-Cook…ppn"])
        detector.start_listening()
        while True:
            detector.wait_for_wakeword()      # blocks until "Hey Cook" heard
            detector.acknowledge()            # clears the event
            … handle command …
        detector.stop()
    """

    def __init__(self, keyword_paths, sensitivity=0.65, device_index=-1):
        """
        Args:
            keyword_paths  (list[str]):          Custom .ppn model paths.
            sensitivity    (float | list[float]): Detection sensitivity 0–1.
            device_index   (int):                 Audio device (-1 = default).
        """
        load_dotenv()
        access_key = os.getenv("PORCUPINE_ACCESS_KEY")
        if not access_key:
            raise ValueError("Missing PORCUPINE_ACCESS_KEY in .env file")

        if isinstance(sensitivity, float):
            sensitivity = [sensitivity] * len(keyword_paths)

        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=keyword_paths,
            sensitivities=sensitivity,
        )
        self.recorder = PvRecorder(
            frame_length=self.porcupine.frame_length,
            device_index=device_index,
        )

        # Threading primitives
        self.detected_event: threading.Event = threading.Event()
        self._stop_event:    threading.Event = threading.Event()
        self._thread: threading.Thread | None = None

        # Most-recent detection metadata
        self.last_keyword_index: int = -1
        self.last_detected_at:   datetime | None = None

        # Optional callback for backwards-compat / extra notifications
        self._on_detect_cb = None

        print(f"[WakeWord] Porcupine v{self.porcupine.version} ready.")

    # ──────────────────────────── Public API ─────────────────────────────────

    def start_listening(self, on_detect=None) -> None:
        """
        Spawn the background listener thread (non-blocking).

        Args:
            on_detect: Optional callback(keyword_index, timestamp) called
                       every time a wake word fires (in addition to setting
                       detected_event).
        """
        if self._thread and self._thread.is_alive():
            print("[WakeWord] Already listening.")
            return

        self._on_detect_cb = on_detect
        self._stop_event.clear()
        self.detected_event.clear()

        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="WakeWord-Listener",
        )
        self._thread.start()
        print("[WakeWord] Background listener started.")

    def wait_for_wakeword(self, timeout: float | None = None) -> bool:
        """
        Block the calling thread until the wake word fires (or timeout).

        Args:
            timeout: Seconds to wait; None = wait forever.

        Returns:
            True if wake word was detected, False if timed out.
        """
        return self.detected_event.wait(timeout=timeout)

    def acknowledge(self) -> None:
        """
        Clear detected_event after the main loop has handled the wake word.
        Must be called before the next wait_for_wakeword().
        """
        self.detected_event.clear()

    def stop(self) -> None:
        """Signal the listener thread to exit and release hardware resources."""
        print("[WakeWord] Stopping listener…")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._cleanup()

    # ──────────────────── Legacy one-shot API (backwards compat) ─────────────

    def start(self, on_detect=None) -> None:
        """
        Original blocking API kept for backwards compatibility.
        Prefer start_listening() + wait_for_wakeword() in new code.
        """
        print("[WakeWord] Listening (blocking mode). Press Ctrl+C to stop.")
        self.recorder.start()
        try:
            while True:
                pcm    = self.recorder.read()
                result = self.porcupine.process(pcm)
                if result >= 0:
                    ts = datetime.now()
                    if on_detect:
                        on_detect(result, ts)
                    else:
                        print(f"[{ts}] Wake word detected! (index={result})")
        except KeyboardInterrupt:
            print("\n[WakeWord] Stopped.")
        finally:
            self._cleanup()

    # ──────────────────────── Internal ───────────────────────────────────────

    def _listen_loop(self) -> None:
        """Background thread: continuously poll Porcupine."""
        print("[WakeWord] Listener thread running…")
        self.recorder.start()
        try:
            while not self._stop_event.is_set():
                pcm    = self.recorder.read()
                result = self.porcupine.process(pcm)

                if result >= 0:
                    ts = datetime.now()
                    self.last_keyword_index = result
                    self.last_detected_at   = ts
                    self.detected_event.set()       # wake up the main loop
                    print(f"[WakeWord] Hey Cook detected at {ts:%H:%M:%S}")

                    if self._on_detect_cb:
                        try:
                            self._on_detect_cb(result, ts)
                        except Exception as exc:
                            print(f"[WakeWord] on_detect callback error: {exc}")
        except Exception as exc:
            if not self._stop_event.is_set():
                print(f"[WakeWord] Listener thread error: {exc}")
        finally:
            self.recorder.stop()
            print("[WakeWord] Listener thread exited.")

    def _cleanup(self) -> None:
        """Release Porcupine + PvRecorder resources."""
        try:
            self.recorder.stop()
            self.recorder.delete()
        except Exception:
            pass
        try:
            self.porcupine.delete()
        except Exception:
            pass


# ──────────────────────── Standalone test ────────────────────────────────────

if __name__ == "__main__":
    detector = WakeWordDetector(
        keyword_paths=["models/Hey-Cook_en_linux_v3_0_0.ppn"],
        sensitivity=0.65,
    )
    detector.start_listening()

    print("Say 'Hey Cook' to test (Ctrl+C to quit)…")
    try:
        while True:
            detected = detector.wait_for_wakeword(timeout=1.0)
            if detected:
                print(f"✓ Wake word confirmed at {detector.last_detected_at:%H:%M:%S}")
                detector.acknowledge()
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()
