"""
Controllable Audio Player using pygame.mixer
Supports pausable/resumable playback with position tracking.
Replaces the previous subprocess.Popen approach which could not be interrupted.
"""

import os
import time
import threading
import logging
from enum import Enum
from typing import Optional, Callable, List, Dict

import pygame

logger = logging.getLogger(__name__)


# ─────────────────────────── State Enum ───────────────────────────────────────

class PlayerState(Enum):
    IDLE     = "idle"
    PLAYING  = "playing"
    PAUSED   = "paused"
    STOPPED  = "stopped"
    FINISHED = "finished"


# ─────────────────────────── AudioPlayer ──────────────────────────────────────

class AudioPlayer:
    """
    A controllable audio player built on pygame.mixer.

    Supports:
      - play()    – start / resume playback of a file
      - pause()   – freeze playback mid-stream
      - resume()  – continue from where it was paused
      - stop()    – stop and reset position
      - seek()    – jump to a position (seconds) in the current track
      - Current-position tracking via a background monitor thread
      - Completion callbacks so callers can chain the next chunk automatically

    Typical usage
    -------------
    >>> player = AudioPlayer()
    >>> player.play("tts_generated_speech/chunk_000.mp3")
    >>> player.pause()
    >>> player.resume()
    >>> player.stop()
    >>> player.shutdown()   # call once, at app exit
    """

    def __init__(self,
                 frequency: int = 44100,
                 buffer_size: int = 2048,
                 on_track_end: Optional[Callable[[], None]] = None):
        """
        Initialise the pygame mixer and the position-tracking thread.

        Args:
            frequency (int):   Audio frequency in Hz (default 44100).
            buffer_size (int): pygame mixer buffer size (default 2048).
            on_track_end:      Optional callback invoked when a track finishes
                               naturally (not when stop() is called explicitly).
        """
        pygame.mixer.pre_init(frequency=frequency, size=-16,
                              channels=2, buffer=buffer_size)
        pygame.mixer.init()

        self._state: PlayerState       = PlayerState.IDLE
        self._current_file: str        = ""
        self._position_sec: float      = 0.0          # playback position in s
        self._play_started_at: float   = 0.0          # wall-clock when last resume
        self._paused_at_sec: float     = 0.0          # position when paused

        self._lock = threading.Lock()
        self._on_track_end: Optional[Callable] = on_track_end

        # Background thread that monitors playback end
        self._monitor_thread = threading.Thread(
            target=self._monitor_playback,
            daemon=True,
            name="AudioPlayer-Monitor"
        )
        self._monitor_running = True
        self._monitor_thread.start()

        logger.info("AudioPlayer initialised (pygame.mixer %s)", pygame.version.ver)

    # ──────────────────────────── Public API ──────────────────────────────────

    def play(self, file_path: str, volume: float = 1.0) -> bool:
        """
        Load and play an audio file from the beginning.
        If a track is currently playing it is stopped first.

        Args:
            file_path (str):  Absolute or relative path to the audio file.
            volume (float):   Playback volume 0.0 – 1.0 (default 1.0).

        Returns:
            bool: True if playback started successfully, False otherwise.
        """
        if not os.path.isfile(file_path):
            logger.error("AudioPlayer.play: file not found – %s", file_path)
            return False

        with self._lock:
            # Stop whatever is playing
            if self._state in (PlayerState.PLAYING, PlayerState.PAUSED):
                pygame.mixer.music.stop()

            try:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
                pygame.mixer.music.play()

                self._current_file  = file_path
                self._position_sec  = 0.0
                self._paused_at_sec = 0.0
                self._play_started_at = time.monotonic()
                self._state = PlayerState.PLAYING

                logger.info("AudioPlayer: playing '%s'", os.path.basename(file_path))
                return True

            except Exception as exc:
                logger.error("AudioPlayer.play error: %s", exc)
                self._state = PlayerState.IDLE
                return False

    def pause(self) -> bool:
        """
        Pause playback. Position is frozen so resume() continues from here.

        Returns:
            bool: True if paused successfully, False if not currently playing.
        """
        with self._lock:
            if self._state != PlayerState.PLAYING:
                logger.warning("AudioPlayer.pause: not playing (state=%s)", self._state.value)
                return False

            pygame.mixer.music.pause()
            elapsed = time.monotonic() - self._play_started_at
            self._paused_at_sec = self._position_sec + elapsed
            self._position_sec  = self._paused_at_sec
            self._state = PlayerState.PAUSED

            logger.info("AudioPlayer: paused at %.2fs", self._paused_at_sec)
            return True

    def resume(self) -> bool:
        """
        Resume playback after a pause().

        Returns:
            bool: True if resumed, False if not in paused state.
        """
        with self._lock:
            if self._state != PlayerState.PAUSED:
                logger.warning("AudioPlayer.resume: not paused (state=%s)", self._state.value)
                return False

            pygame.mixer.music.unpause()
            self._play_started_at = time.monotonic()
            self._state = PlayerState.PLAYING

            logger.info("AudioPlayer: resumed from %.2fs", self._paused_at_sec)
            return True

    def stop(self) -> bool:
        """
        Stop playback and reset position to 0.

        Returns:
            bool: Always True.
        """
        with self._lock:
            pygame.mixer.music.stop()
            self._position_sec  = 0.0
            self._paused_at_sec = 0.0
            self._current_file  = ""
            self._state = PlayerState.STOPPED
            logger.info("AudioPlayer: stopped")
            return True

    def seek(self, position_sec: float) -> bool:
        """
        Seek to a given position in the currently loaded track.
        Note: pygame only supports seeking for certain formats (OGG, MP3 ≥ pygame 2.x).

        Args:
            position_sec (float): Target position in seconds.

        Returns:
            bool: True if seek succeeded, False otherwise.
        """
        with self._lock:
            if not self._current_file:
                logger.warning("AudioPlayer.seek: no file loaded")
                return False

            was_playing = self._state == PlayerState.PLAYING
            was_paused  = self._state == PlayerState.PAUSED

            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.load(self._current_file)
                pygame.mixer.music.play(start=position_sec)

                self._position_sec  = position_sec
                self._paused_at_sec = position_sec
                self._play_started_at = time.monotonic()
                self._state = PlayerState.PLAYING

                if was_paused:
                    pygame.mixer.music.pause()
                    self._state = PlayerState.PAUSED

                logger.info("AudioPlayer: seeked to %.2fs", position_sec)
                return True

            except Exception as exc:
                logger.error("AudioPlayer.seek error: %s", exc)
                return False

    def set_volume(self, volume: float) -> None:
        """
        Adjust playback volume without stopping.

        Args:
            volume (float): 0.0 (silent) – 1.0 (full volume).
        """
        pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
        logger.debug("AudioPlayer: volume set to %.2f", volume)

    def shutdown(self) -> None:
        """
        Release pygame mixer resources. Call once when the application exits.
        """
        self._monitor_running = False
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        logger.info("AudioPlayer: mixer shut down")

    # ──────────────────────── Property accessors ──────────────────────────────

    @property
    def state(self) -> PlayerState:
        """Current PlayerState enum value."""
        return self._state

    @property
    def is_playing(self) -> bool:
        """True while audio is actively playing (not paused, not stopped)."""
        return self._state == PlayerState.PLAYING

    @property
    def is_paused(self) -> bool:
        """True while audio is paused mid-track."""
        return self._state == PlayerState.PAUSED

    @property
    def is_idle(self) -> bool:
        """True when no track is loaded / playback has never started."""
        return self._state in (PlayerState.IDLE, PlayerState.STOPPED, PlayerState.FINISHED)

    @property
    def current_file(self) -> str:
        """Path to the currently loaded audio file (empty string if none)."""
        return self._current_file

    @property
    def position(self) -> float:
        """
        Approximate current playback position in seconds.
        Accurate while playing; frozen when paused.
        """
        with self._lock:
            if self._state == PlayerState.PLAYING:
                return self._position_sec + (time.monotonic() - self._play_started_at)
            return self._position_sec

    # ──────────────────────── Background monitor ──────────────────────────────

    def _monitor_playback(self) -> None:
        """
        Background daemon thread.
        Detects when pygame.mixer.music finishes naturally and transitions
        state to FINISHED, then fires the optional on_track_end callback.
        """
        while self._monitor_running:
            time.sleep(0.1)
            with self._lock:
                if self._state != PlayerState.PLAYING:
                    continue
                # pygame.mixer.music.get_busy() returns False when track ends
                if not pygame.mixer.music.get_busy():
                    elapsed = time.monotonic() - self._play_started_at
                    self._position_sec += elapsed
                    self._state = PlayerState.FINISHED
                    logger.info("AudioPlayer: track finished naturally")

            # Fire callback outside the lock to avoid dead-locks
            if self._state == PlayerState.FINISHED and self._on_track_end:
                try:
                    self._on_track_end()
                except Exception as exc:
                    logger.error("AudioPlayer on_track_end callback error: %s", exc)


# ─────────────────────── ChunkedAudioPlayer ───────────────────────────────────

class ChunkedAudioPlayer:
    """
    Higher-level player that consumes the List[Dict] chunk metadata produced by
    RecipeTTS.generate_speech_chunks() and plays them sequentially.

    Supports:
      - play_chunks()           – play all chunks in order
      - skip_to_chunk()         – jump directly to any chunk index
      - pause() / resume()      – delegate to underlying AudioPlayer
      - stop()                  – stop and reset queue
      - current_chunk_index     – which chunk is playing right now
      - progress()              – fraction completed (0.0 – 1.0)

    Interrupt / new-command pattern
    --------------------------------
    Call stop() to immediately halt playback; the chunk queue is cleared.
    Call play_chunks() again with a new list to start fresh.
    """

    def __init__(self, player: Optional[AudioPlayer] = None):
        """
        Args:
            player (AudioPlayer, optional): Re-use an existing AudioPlayer instance.
                                            Creates a new one if not provided.
        """
        self._player = player or AudioPlayer(on_track_end=self._on_chunk_finished)
        # If caller provided their own player, override its callback
        if player is not None:
            self._player._on_track_end = self._on_chunk_finished

        self._chunks: List[Dict]         = []
        self._current_index: int         = -1
        self._playing_lock               = threading.Lock()
        self._stop_requested: bool       = False

        # Callbacks
        self._on_chunk_start:  Optional[Callable[[int, Dict], None]] = None
        self._on_chunk_end:    Optional[Callable[[int, Dict], None]] = None
        self._on_all_finished: Optional[Callable[[], None]]          = None

    # ─────────────────────── Public API ───────────────────────────────────────

    def set_callbacks(self,
                      on_chunk_start:  Optional[Callable[[int, Dict], None]] = None,
                      on_chunk_end:    Optional[Callable[[int, Dict], None]] = None,
                      on_all_finished: Optional[Callable[[], None]]          = None) -> None:
        """
        Register optional event callbacks.

        Args:
            on_chunk_start(index, chunk_dict):  fired just before a chunk starts playing.
            on_chunk_end(index, chunk_dict):    fired just after a chunk finishes.
            on_all_finished():                  fired when the entire queue is done.
        """
        self._on_chunk_start  = on_chunk_start
        self._on_chunk_end    = on_chunk_end
        self._on_all_finished = on_all_finished

    def play_chunks(self, chunks: List[Dict], start_index: int = 0) -> bool:
        """
        Start playing a list of chunk dicts (as returned by RecipeTTS).

        Args:
            chunks (List[Dict]):  Each dict must have at least {"audio_path": "...", "text": "..."}.
            start_index (int):    Chunk to begin from (default 0).

        Returns:
            bool: True if the first chunk started playing, False otherwise.
        """
        valid_chunks = [c for c in chunks if c.get("audio_path") and not c.get("error")]
        if not valid_chunks:
            logger.error("ChunkedAudioPlayer: no valid chunks to play")
            return False

        with self._playing_lock:
            self._chunks         = valid_chunks
            self._stop_requested = False
            self._current_index  = -1

        return self._play_index(start_index)

    def skip_to_chunk(self, index: int) -> bool:
        """
        Stop the current chunk and jump directly to chunk at *index*.

        Args:
            index (int): Target chunk index (0-based).

        Returns:
            bool: True if the target chunk started, False if index out of range.
        """
        if not self._chunks or index >= len(self._chunks):
            logger.error("ChunkedAudioPlayer.skip_to_chunk: index %d out of range", index)
            return False

        self._player.stop()
        return self._play_index(index)

    def pause(self) -> bool:
        """Pause the currently playing chunk."""
        return self._player.pause()

    def resume(self) -> bool:
        """Resume the currently paused chunk."""
        return self._player.resume()

    def add_chunk(self, chunk: Dict) -> None:
        """
        Append *chunk* to the live queue.

        If the player is currently idle (all previously queued chunks have
        finished playing), playback is restarted immediately on this chunk.
        Safe to call from any thread while playback is ongoing.

        Args:
            chunk (Dict): Must have at least ``{"audio_path": "...", "text": "..."}``.
                          Chunks with ``error=True`` or missing ``audio_path`` are
                          silently ignored.
        """
        if not chunk.get("audio_path") or chunk.get("error"):
            return
        with self._playing_lock:
            if self._stop_requested:
                return
            self._chunks.append(chunk)
            new_index = len(self._chunks) - 1
            # Kick off playback only if the player is sitting idle
            should_start = self._player.state in (
                PlayerState.IDLE, PlayerState.STOPPED, PlayerState.FINISHED
            ) and self._current_index < new_index

        if should_start:
            self._play_index(new_index)

    def stop(self) -> bool:
        """Stop playback and clear the chunk queue."""
        with self._playing_lock:
            self._stop_requested = True
            self._chunks         = []
            self._current_index  = -1
        self._player.stop()
        logger.info("ChunkedAudioPlayer: stopped & queue cleared")
        return True

    def wait_until_done(self, poll_interval: float = 0.2) -> None:
        """
        Block the calling thread until all chunks have finished playing
        or stop() has been called.  Useful in scripts / tests.

        Args:
            poll_interval (float): Seconds between busy-wait polls.
        """
        while True:
            with self._playing_lock:
                done = self._stop_requested or (
                    self._current_index >= len(self._chunks) - 1
                    and self._player.state in (
                        PlayerState.FINISHED,
                        PlayerState.STOPPED,
                        PlayerState.IDLE,
                    )
                )
            if done:
                break
            time.sleep(poll_interval)

    # ─────────────────────── Properties ───────────────────────────────────────

    @property
    def current_chunk_index(self) -> int:
        """0-based index of the chunk currently playing; -1 if idle."""
        return self._current_index

    @property
    def current_chunk(self) -> Optional[Dict]:
        """Metadata dict of the chunk currently playing, or None."""
        if 0 <= self._current_index < len(self._chunks):
            return self._chunks[self._current_index]
        return None

    @property
    def total_chunks(self) -> int:
        """Total number of chunks in the current queue."""
        return len(self._chunks)

    @property
    def state(self) -> PlayerState:
        """Delegate to the underlying AudioPlayer state."""
        return self._player.state

    @property
    def is_playing(self) -> bool:
        return self._player.is_playing

    @property
    def is_paused(self) -> bool:
        return self._player.is_paused

    def progress(self) -> Dict:
        """
        Return a progress snapshot.

        Returns:
            Dict with keys: current_index, total_chunks, played_chunks,
                            remaining_chunks, progress_percentage, current_text.
        """
        total   = len(self._chunks)
        played  = max(0, self._current_index)   # chunks fully played
        remaining = max(0, total - self._current_index - 1)
        pct = (played / total * 100) if total > 0 else 0.0

        return {
            "current_index":      self._current_index,
            "total_chunks":       total,
            "played_chunks":      played,
            "remaining_chunks":   remaining,
            "progress_percentage": round(pct, 1),
            "current_text":       self.current_chunk.get("text", "") if self.current_chunk else "",
        }

    # ─────────────────────── Internal helpers ─────────────────────────────────

    def _play_index(self, index: int) -> bool:
        """Start playback for chunk at *index*."""
        with self._playing_lock:
            if self._stop_requested or index >= len(self._chunks):
                return False
            self._current_index = index

        chunk = self._chunks[index]
        logger.info("ChunkedAudioPlayer: playing chunk %d/%d – '%s'",
                    index + 1, len(self._chunks), chunk.get("text", "")[:60])

        if self._on_chunk_start:
            try:
                self._on_chunk_start(index, chunk)
            except Exception as exc:
                logger.error("on_chunk_start callback error: %s", exc)

        return self._player.play(chunk["audio_path"])

    def _on_chunk_finished(self) -> None:
        """Called by AudioPlayer when the current chunk ends naturally."""
        finished_index = self._current_index

        if self._on_chunk_end and 0 <= finished_index < len(self._chunks):
            try:
                self._on_chunk_end(finished_index, self._chunks[finished_index])
            except Exception as exc:
                logger.error("on_chunk_end callback error: %s", exc)

        # Advance to next chunk
        next_index = finished_index + 1
        with self._playing_lock:
            if self._stop_requested or next_index >= len(self._chunks):
                # Queue exhausted
                if not self._stop_requested and self._on_all_finished:
                    try:
                        self._on_all_finished()
                    except Exception as exc:
                        logger.error("on_all_finished callback error: %s", exc)
                return

        self._play_index(next_index)


# ─────────────────────────────── Quick test ───────────────────────────────────

def _test_audio_player():
    """
    Minimal smoke-test.  Requires at least one .mp3 file in tts_generated_speech/.
    Run:  python modules/audio_player.py
    """
    import glob

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    )

    print("=" * 60)
    print("AUDIO PLAYER TEST")
    print("=" * 60)

    # ── 1. Basic AudioPlayer ──────────────────────────────────────
    mp3_files = sorted(glob.glob("tts_generated_speech/*.mp3"))
    if not mp3_files:
        print("\n⚠  No .mp3 files found in tts_generated_speech/. Skipping playback tests.")
        print("   (Generate some audio first with RecipeTTS, then re-run this test.)")
        return

    player = AudioPlayer()
    print(f"\n1. Playing: {mp3_files[0]}")
    player.play(mp3_files[0])

    time.sleep(1.0)
    print(f"   Position: {player.position:.2f}s  State: {player.state.value}")

    print("2. Pausing...")
    player.pause()
    print(f"   State: {player.state.value}  Position frozen at: {player.position:.2f}s")

    time.sleep(5.0)
    print("3. Resuming...")
    player.resume()
    print(f"   State: {player.state.value}")

    time.sleep(1.0)
    print("4. Stopping...")
    player.stop()
    print(f"   State: {player.state.value}")

    # ── 2. ChunkedAudioPlayer ─────────────────────────────────────
    if len(mp3_files) >= 2:
        print("\n5. ChunkedAudioPlayer – playing first 2 chunks sequentially")
        fake_chunks = [
            {"chunk_index": i, "text": f"chunk {i}", "audio_path": mp3_files[i], "played": False}
            for i in range(min(7, len(mp3_files)))
        ]

        chunked = ChunkedAudioPlayer(player=player)
        chunked.set_callbacks(
            on_chunk_start=lambda i, c: print(f"   ▶ Chunk {i} started: {c['text']}"),
            on_chunk_end=lambda i, c:   print(f"   ■ Chunk {i} ended"),
            on_all_finished=lambda:     print("   ✓ All chunks finished"),
        )
        chunked.play_chunks(fake_chunks)
        chunked.wait_until_done()
    else:
        print("\n(Only 1 mp3 found; skipping ChunkedAudioPlayer test)")

    player.shutdown()
    print("\n✓ Audio player tests complete")


if __name__ == "__main__":
    _test_audio_player()

