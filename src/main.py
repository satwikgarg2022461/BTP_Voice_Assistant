#!/usr/bin/env python3
"""
Voice Assistant — main entry point
===================================

Architecture
------------

  ┌─────────────────────────────────────────────────┐
  │  WakeWord thread  (daemon, always running)       │
  │  PvRecorder → Porcupine → detected_event.set()  │
  └────────────────────────┬────────────────────────┘
                           │ detected_event
                           ▼
  ┌─────────────────────────────────────────────────┐
  │  Main loop                                       │
  │  1. wait_for_wakeword()                          │
  │  2. pause current AudioPlayer                    │
  │  3. record + ASR + intent classify               │
  │  4. dispatch intent → navigator / LLM / small-  │
  │     talk / recipe-search                         │
  │  5. TTS-chunk → ChunkedAudioPlayer               │
  │     (resumes previous audio if no command given) │
  └─────────────────────────────────────────────────┘

Interrupt contract
------------------
  • WakeWord thread NEVER stops — it listens even while TTS is playing.
  • On detection: current playback is PAUSED immediately.
  • If user gives a valid command  → stop old playback, start new response.
  • If user gives no useful input  → resume previous playback from where it
    was paused.
"""

import os
import sys
import time
import argparse
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.wakeword        import WakeWordDetector
from modules.vad             import ShortRecorder
from modules.deepgram_asr    import DeepgramASR
from modules.asr             import WhisperASR
from modules.intent_classifier import IntentClassifier, Intent
from modules.navigator       import RecipeNavigator
from modules.session_manager import SessionManager
from modules.retriever       import RecipeRetriever
from modules.llm             import RecipeLLM
from modules.tts             import RecipeTTS, SarvamTTS, get_tts_engine
from modules.audio_player    import ChunkedAudioPlayer, AudioPlayer


# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _timer(label: str, store: dict | None = None):
    """Context-manager that prints elapsed time for *label* and optionally
    stores the duration (seconds) in *store[label]*."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"[⏱  {label}] {elapsed * 1000:.1f} ms")
    if store is not None:
        store[label] = elapsed

class VoiceAssistant:
    """
    Coordinates all modules for a conversational recipe voice assistant.

    Key invariant
    -------------
    self._player   : ChunkedAudioPlayer  – the single active player.
                     Replaced (not mutated) whenever a new response starts.
    self._session  : dict                – in-memory mirror of Redis session.
                     Persisted to Redis after every interaction.
    """

    WAKE_MODEL  = "models/Hey-Cook_en_linux_v3_0_0.ppn"
    SESSION_ID  = "user_001"          # single-user prototype
    LISTEN_TIMEOUT = 8.0              # seconds to wait for user speech after wake

    def __init__(
        self,
        keyword_paths   = None,
        wake_sensitivity = 0.65,
        device_index    = -1,
        recordings_dir  = "voice_recordings",
        asr_type        = "api",       # "api" = Deepgram, "local" = Whisper
        asr_model       = "small",
        asr_text_dir    = "ASR_text",
        db_path         = "recipes_demo.db",
        collection_name = "recipes_collection",
        food_dict_path  = "data/food_dictionary.csv",
        fuzzy_score_cutoff = 70,
        tts_provider    = "deepgram",  # "deepgram" or "sarvam"
        autocorrect     = True,        # set False to skip fuzzy ASR correction
    ):
        print("=" * 60)
        print("  Initialising Voice Assistant")
        print("=" * 60)

        if keyword_paths is None:
            keyword_paths = [self.WAKE_MODEL]

        # ── directories ──────────────────────────────────────────────────────
        self.recordings_dir = recordings_dir
        self.asr_text_dir   = asr_text_dir
        os.makedirs(recordings_dir, exist_ok=True)
        os.makedirs(asr_text_dir,   exist_ok=True)

        # ── wake word ─────────────────────────────────────────────────────────
        self.wake_detector = WakeWordDetector(
            keyword_paths=keyword_paths,
            sensitivity=wake_sensitivity,
            device_index=device_index,
        )

        # ── VAD recorder ─────────────────────────────────────────────────────
        self.recorder = ShortRecorder(
            sample_rate=16000,
            frame_length=512,
            pre_roll_secs=1.0,
            silence_thresh=1000,
            silence_duration=1.0,
        )

        # ── ASR ───────────────────────────────────────────────────────────────
        self.asr_type = asr_type
        if asr_type == "api":
            print("ASR  : Deepgram API")
            self.asr           = DeepgramASR()
            self.asr_corrector = WhisperASR(model_size="base")
        else:
            print(f"ASR  : Whisper ({asr_model})")
            self.asr           = WhisperASR(model_size=asr_model)
            self.asr_corrector = self.asr

        # ── intent classifier ────────────────────────────────────────────────
        self.classifier = IntentClassifier(
            confidence_threshold=0.70,
            use_llm_fallback=True,
        )

        # ── navigator ────────────────────────────────────────────────────────
        self.navigator = RecipeNavigator()

        # ── session manager ──────────────────────────────────────────────────
        self.session_mgr = SessionManager(session_ttl=3600)
        self._session: Optional[dict] = None          # in-memory mirror
        self._session_lock = threading.Lock()

        # ── retriever ────────────────────────────────────────────────────────
        self.retriever = RecipeRetriever(
            db_path=db_path,
            collection_name=collection_name,
        )

        # ── LLM ──────────────────────────────────────────────────────────────
        try:
            self.llm = RecipeLLM()
            print("LLM  : OK")
        except Exception as e:
            print(f"LLM  : unavailable ({e})")
            self.llm = None

        # ── TTS ───────────────────────────────────────────────────────────────
        try:
            self.tts = get_tts_engine(tts_provider)
            print(f"TTS  : OK ({tts_provider})")
        except Exception as e:
            print(f"TTS  : unavailable ({e})")
            self.tts = None

        # ── audio player ─────────────────────────────────────────────────────
        # One shared low-level player; ChunkedAudioPlayer wraps it.
        self._base_player  = AudioPlayer()
        self._player: Optional[ChunkedAudioPlayer] = None
        self._player_lock  = threading.Lock()

        # ── fuzzy-matching vocabulary ────────────────────────────────────────
        self.autocorrect        = autocorrect
        self.fuzzy_score_cutoff = fuzzy_score_cutoff
        self.recipe_names: list = []
        self.ingredients:  list = []
        if autocorrect and os.path.exists(food_dict_path):
            self.recipe_names, self.ingredients = \
                self.asr_corrector.load_recipe_terms(food_dict_path)
            print(f"Vocab : {len(self.recipe_names)} recipes, "
                  f"{len(self.ingredients)} ingredients loaded")
        elif not autocorrect:
            print("Vocab : autocorrect disabled — fuzzy matching skipped")

        print("\n✓ Voice Assistant ready — say 'Hey Cook' to start!\n")

    # ─────────────────────────── Main run loop ───────────────────────────────

    def run(self) -> None:
        """Start the wake-word thread then enter the coordination loop."""

        # Wake word detector runs as a daemon thread permanently
        self.wake_detector.start_listening()

        print("Listening for 'Hey Cook'…  (Ctrl+C to exit)\n")
        try:
            while True:
                # Block until Porcupine fires
                self.wake_detector.wait_for_wakeword()
                self.wake_detector.acknowledge()      # clear event for next cycle

                self._handle_wakeword_interrupt()

        except KeyboardInterrupt:
            print("\n[Main] Shutting down…")
        finally:
            self._shutdown()

    # ───────────────────── Interrupt handler (main thread) ───────────────────

    def _handle_wakeword_interrupt(self) -> None:
        """
        Called every time 'Hey Cook' is detected.

        Steps
        -----
        1. Pause current audio immediately.
        2. Record user command with VAD.
        3. ASR + text correction.
        4. Intent classify.
        5. Dispatch:
             • navigation intents  → navigator
             • search_recipe       → retriever + LLM
             • question            → RAG + LLM
             • small_talk          → LLM
             • stop_pause          → stay paused
             • resume              → resume previous audio
             • no speech / unknown → resume previous audio
        6. For new responses: generate TTS chunks, hand to ChunkedAudioPlayer.
        """
        timings: dict = {}          # collects step durations for the summary
        interaction_start = time.perf_counter()

        # ── 1. Pause current playback ─────────────────────────────────────
        had_audio = self._pause_playback()
        print("[Main] ▶ Wake word! Audio paused." if had_audio
              else "[Main] ▶ Wake word!")

        # ── 2. VAD + Record command ───────────────────────────────────────
        ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_path = os.path.join(self.recordings_dir, f"recording_{ts_str}.wav")
        with _timer("VAD / Recording", timings):
            audio_path = self.recorder.record_once(rec_path)
        print(f"[Main] Recorded → {audio_path}")

        # ── 3. Transcribe ─────────────────────────────────────────────────
        text = self._transcribe(audio_path, ts_str, timings)
        if not text:
            print("[Main] No speech detected — resuming previous audio.")
            self._resume_or_pass()
            return

        print(f"[Main] Heard: '{text}'")

        # ── 4. Classify intent ────────────────────────────────────────────
        context = {}
        with self._session_lock:
            if self._session:
                context = {
                    "current_state": self._session.get("current_state", ""),
                    "recipe_title":  self._session.get("recipe_title", ""),
                }

        with _timer("Intent Classifier", timings):
            intent, confidence, entities = self.classifier.classify(text, context)
        print(f"[Main] Intent={intent.value}  conf={confidence:.2f}  "
              f"entities={entities}")

        # ── 5. Dispatch ───────────────────────────────────────────────────
        self._dispatch(intent, text, entities, had_audio, timings)

        # ── 6. Timing summary ─────────────────────────────────────────────
        total = time.perf_counter() - interaction_start
        print("\n" + "─" * 52)
        print(f"{'Pipeline Timing Summary':^52}")
        print("─" * 52)
        for step, duration in timings.items():
            print(f"  {step:<30} {duration * 1000:>8.1f} ms")
        print("─" * 52)
        print(f"  {'TOTAL':<30} {total * 1000:>8.1f} ms")
        print("─" * 52 + "\n")

    # ─────────────────────────── Dispatch ────────────────────────────────────

    def _dispatch(self, intent: Intent, text: str,
                  entities: dict, had_audio: bool,
                  timings: dict | None = None) -> None:
        """Route intent to the correct handler."""

        t = timings if timings is not None else {}

        # ── navigation ───────────────────────────────────────────────────
        if intent in (Intent.NAV_NEXT, Intent.NAV_PREV, Intent.NAV_GO_TO,
                      Intent.NAV_REPEAT, Intent.NAV_REPEAT_INGREDIENTS,
                      Intent.NAV_START):
            self._handle_navigation(intent, entities, t)

        # ── recipe search ────────────────────────────────────────────────
        elif intent == Intent.SEARCH_RECIPE:
            self._handle_recipe_search(text, entities, t)

        # ── start cooking ────────────────────────────────────────────────
        elif intent == Intent.START_RECIPE:
            self._handle_start_recipe(t)

        # ── question (RAG + LLM) ─────────────────────────────────────────
        elif intent == Intent.QUESTION:
            self._handle_question(text, t)

        # ── small talk ───────────────────────────────────────────────────
        elif intent == Intent.SMALL_TALK:
            self._handle_small_talk(text, t)

        # ── pause / stop ─────────────────────────────────────────────────
        elif intent == Intent.STOP_PAUSE:
            print("[Main] Staying paused as requested.")
            # Audio is already paused; do nothing

        # ── resume ───────────────────────────────────────────────────────
        elif intent == Intent.RESUME:
            self._resume_or_pass()

        # ── confirm / cancel ─────────────────────────────────────────────
        elif intent == Intent.CONFIRM:
            self._handle_confirm()

        elif intent == Intent.CANCEL:
            self._handle_cancel()

        # ── help ─────────────────────────────────────────────────────────
        elif intent == Intent.HELP:
            self._speak_text(
                "You can say: next step, previous step, repeat, repeat ingredients, "
                "go to step 3, restart, pause, resume, or ask me any question "
                "about the recipe.",
                t,
            )

        # ── unknown / no-op → resume ─────────────────────────────────────
        else:
            print("[Main] Unknown intent — resuming previous audio.")
            self._resume_or_pass()

    # ──────────────────────── Navigation handlers ────────────────────────────

    def _handle_navigation(self, intent: Intent, entities: dict,
                           timings: dict | None = None) -> None:
        """Resolve the nav intent → NavigationResult → speak the text."""
        with self._session_lock:
            session = self._session

        if session is None:
            self._speak_text("No recipe is active. Please search for a recipe first.")
            return

        nav   = self.navigator
        result = None

        if intent == Intent.NAV_NEXT:
            result = nav.get_next_step(session)

        elif intent == Intent.NAV_PREV:
            result = nav.get_previous_step(session)

        elif intent == Intent.NAV_GO_TO:
            target = entities.get("step_number") or entities.get("step_text", "")
            result = nav.jump_to_step(session, target)

        elif intent == Intent.NAV_REPEAT:
            result = nav.get_current_step(session)

        elif intent == Intent.NAV_REPEAT_INGREDIENTS:
            result = nav.get_current_ingredients(session)

        elif intent == Intent.NAV_START:
            result = nav.restart(session)

        if result is None:
            return

        # Update in-memory session
        with self._session_lock:
            nav.update_session_from_result(self._session, result)
            session_id = self._session.get("session_id", self.SESSION_ID)
            step_index = self._session.get("step_index", 0)

        # Persist to Redis
        self.session_mgr.update_session(session_id, {
            "step_index":     step_index,
            "chunk_index":    result.chunk_index,
            "current_section": result.section,
            "last_intent":    result.intent,
        })

        # Log to conversation history
        self.session_mgr.add_conversation_turn(
            session_id, role="assistant",
            content=result.text, intent=result.intent,
        )

        # Mark step spoken in session
        if result.section == "steps" and result.step_index > 0:
            self.session_mgr.mark_step_spoken(session_id, result.step_index)
        elif result.section == "ingredients":
            self.session_mgr.mark_section_spoken(session_id, "ingredients")

        # Speak the navigation result
        self._stop_playback()
        self._speak_text(result.text, timings)

    # ──────────────────────── Recipe search ──────────────────────────────────

    def _handle_recipe_search(self, text: str, entities: dict,
                              timings: dict | None = None) -> None:
        """Search for a recipe, create a session, and present ingredients."""
        print("[Main] Searching for recipe…")

        with _timer("RAG / Retrieval", timings):
            results = self.retriever.search_recipes(text, limit=1)
        if not results:
            self._speak_text("I couldn't find a matching recipe. Please try again.",
                             timings)
            return

        top = results[0]
        recipe_id    = top.get("recipe_id")
        # Use recipe title from navigator data (clean) — not text_preview which
        # contains the raw ingredient dump from the vector store.
        recipe_data = self.navigator.load_recipe(recipe_id)
        if recipe_data is None:
            recipe_title = top.get("title", "the recipe")
            self._speak_text(f"I found {recipe_title} but couldn't load its steps.",
                             timings)
            return
        recipe_title = recipe_data.title

        # Create / overwrite session
        nav_fields = self.navigator.build_session_nav_fields(recipe_data)
        session = self.session_mgr.create_session(
            session_id    = self.SESSION_ID,
            recipe_id     = str(recipe_id),
            recipe_title  = recipe_title,
            total_steps   = recipe_data.total_steps,
        )
        session.update(nav_fields)
        session["session_id"] = self.SESSION_ID

        with self._session_lock:
            self._session = session

        response = (
            f"Great! I found {recipe_title}. "
            f"It has {recipe_data.total_steps} steps. "
            f"Say 'start cooking' to begin, or ask me anything."
        )
        self._stop_playback()
        self._speak_text(response, timings)

    # ──────────────────────── Start recipe ───────────────────────────────────

    def _handle_start_recipe(self, timings: dict | None = None) -> None:
        """
        Begin a recipe: read the ingredients list, then immediately move to
        step 1 so the user hears the full recipe flow without an extra command.
        """
        with self._session_lock:
            session = self._session

        if session is None:
            self._speak_text("No recipe loaded. Please search for a recipe first.")
            return

        # Read ingredients
        ingr_result = self.navigator.get_current_ingredients(session)
        with self._session_lock:
            self.navigator.update_session_from_result(self._session, ingr_result)

        # Advance to step 1
        step_result = self.navigator.get_next_step(self._session)
        with self._session_lock:
            self.navigator.update_session_from_result(self._session, step_result)
            session_id = self._session.get("session_id", self.SESSION_ID)

        # Persist step 1 to Redis
        self.session_mgr.update_session(session_id, {
            "step_index":      step_result.step_index,
            "chunk_index":     step_result.chunk_index,
            "current_section": step_result.section,
            "last_intent":     "start_recipe",
        })
        self.session_mgr.mark_section_spoken(session_id, "ingredients")
        self.session_mgr.mark_step_spoken(session_id, step_result.step_index)

        # Speak ingredients + step 1 + prompt as one continuous response
        prompt_text = "Say 'Hey Cook, next step' whenever you're ready to continue."
        combined = ingr_result.text + " " + step_result.text + " " + prompt_text
        self._stop_playback()
        self._speak_text(combined, timings)

    # ──────────────────────── Question (RAG + LLM) ───────────────────────────

    def _handle_question(self, text: str,
                         timings: dict | None = None) -> None:
        """Answer a recipe-related question using retriever + LLM context."""
        if self.llm is None:
            self._speak_text("I can't answer questions right now. The LLM is unavailable.")
            return

        with self._session_lock:
            recipe_title = (self._session or {}).get("recipe_title", "")
            history = (self._session or {}).get("conversation_history", [])

        print("[Main] Answering question with RAG+LLM…")
        with _timer("RAG / Retrieval", timings):
            results = self.retriever.search_recipes(text, limit=1)

        # Build recipe context string for RAG
        recipe_context = ""
        if results:
            top = results[0]
            recipe_context = top.get("text_preview", top.get("title", ""))

        with _timer("LLM Generation", timings):
            answer = self.llm.answer_recipe_question(
                text,
                recipe_context=recipe_context,
                conversation_history=history,
            )

        # If LLM returns structured JSON, extract spoken form
        spoken = self._extract_spoken_text(answer)
        self._stop_playback()
        self._speak_text(spoken, timings)

        # Update conversation history
        with self._session_lock:
            sid = (self._session or {}).get("session_id", self.SESSION_ID)
        self.session_mgr.add_conversation_turn(
            sid, role="user", content=text, intent="question"
        )
        self.session_mgr.add_conversation_turn(
            sid, role="assistant", content=spoken, intent="question"
        )

    # ──────────────────────── Small talk ─────────────────────────────────────

    def _handle_small_talk(self, text: str,
                            timings: dict | None = None) -> None:
        """Send small-talk to the LLM and speak the reply."""
        if self.llm is None:
            self._speak_text("I'm your cooking assistant! Let me know when you'd "
                             "like to start or continue the recipe.")
            return

        print("[Main] Small talk → LLM…")
        with self._session_lock:
            history = (self._session or {}).get("conversation_history", [])
            context = {
                "recipe_title":    (self._session or {}).get("recipe_title", ""),
                "step_index":      (self._session or {}).get("step_index", 0),
                "current_section": (self._session or {}).get("current_section", ""),
                "paused":          (self._session or {}).get("paused", False),
            }

        with _timer("LLM Generation", timings):
            answer = self.llm.generate_conversational_response(
                text, "small_talk",
                conversation_history=history,
                context=context,
            )
        spoken = answer if isinstance(answer, str) else self._extract_spoken_text(answer)
        self._stop_playback()
        self._speak_text(spoken, timings)

    # ──────────────────────── Confirm / Cancel ───────────────────────────────

    def _handle_confirm(self) -> None:
        """User said yes — advance to next step if in a recipe."""
        with self._session_lock:
            session = self._session
        if session and session.get("current_state") in ("READING_STEPS",
                                                         "READING_INGREDIENTS"):
            self._handle_navigation(Intent.NAV_NEXT, {})
        else:
            self._speak_text("Okay!")

    def _handle_cancel(self) -> None:
        """User said no / cancel — stay put."""
        self._speak_text("Okay, no problem. Just say 'Hey Cook' whenever you're ready.")

    # ─────────────────────── Audio player helpers ────────────────────────────

    def _pause_playback(self) -> bool:
        """Pause the ChunkedAudioPlayer if it is currently playing."""
        with self._player_lock:
            if self._player and self._player.is_playing:
                self._player.pause()
                return True
        return False

    def _resume_or_pass(self) -> None:
        """Resume previous audio if paused; otherwise do nothing."""
        with self._player_lock:
            if self._player and self._player.is_paused:
                self._player.resume()
                print("[Main] ▶ Resumed previous audio.")
            else:
                print("[Main] Nothing to resume.")

    def _stop_playback(self) -> None:
        """Stop and discard the current player (before starting a new response)."""
        with self._player_lock:
            if self._player:
                self._player.stop()
                self._player = None

    def _speak_text(self, text: str, timings: dict | None = None) -> None:
        """
        Generate TTS chunks for *text* and start playing them via a new
        ChunkedAudioPlayer.

        Streaming model
        ---------------
        A background thread calls ``generate_speech_streaming()`` which yields
        each chunk as soon as the Deepgram API returns it.  The first chunk is
        fed to the player immediately (user hears audio after ~1 API RTT, not
        after all chunks finish).  Subsequent chunks are enqueued on-the-fly.

        Returns immediately; both generation and playback run in background threads.
        """
        if not self.tts:
            print(f"[TTS unavailable] {text}")
            return

        if not text.strip():
            return

        ts_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[Main] TTS streaming generation started…")

        # Create player up-front so background thread can call add_chunk on it
        chunk_start_times: dict[int, float] = {}

        with self._player_lock:
            player = ChunkedAudioPlayer(player=self._base_player)

            def _on_chunk_start(i: int, c: dict) -> None:
                chunk_start_times[i] = time.perf_counter()
                print(
                    f"[Audio] ▶ chunk {i+1}: "
                    f"{c['text'][:60]}…"
                )

            def _on_chunk_end(i: int, c: dict) -> None:
                if i in chunk_start_times:
                    dur = time.perf_counter() - chunk_start_times[i]
                    print(f"[⏱  Audio chunk {i+1}] {dur * 1000:.1f} ms")

            player.set_callbacks(
                on_chunk_start  = _on_chunk_start,
                on_chunk_end    = _on_chunk_end,
                on_all_finished = lambda: print("[Audio] ✓ Playback complete."),
            )
            self._player = player

        t_stream_start = time.perf_counter()
        first_chunk_timed = False

        def _stream_and_feed() -> None:
            nonlocal first_chunk_timed
            for chunk in self.tts.generate_speech_streaming(
                text, output_prefix=f"response_{ts_prefix}"
            ):
                if not first_chunk_timed:
                    elapsed = time.perf_counter() - t_stream_start
                    print(f"[⏱  TTS First Chunk] {elapsed * 1000:.1f} ms")
                    if timings is not None:
                        timings["TTS First Chunk"] = elapsed
                    first_chunk_timed = True
                player.add_chunk(chunk)

        threading.Thread(target=_stream_and_feed, daemon=True,
                         name="TTS-StreamFeed").start()

    # ─────────────────────────── Helpers ─────────────────────────────────────

    def _transcribe(self, audio_path: str, ts_str: str,
                    timings: dict | None = None) -> str:
        """Run ASR on *audio_path*, apply fuzzy correction, return clean text."""
        try:
            with _timer("Speech-to-Text (ASR)", timings):
                result = self.asr.transcribe_audio(audio_path)
            text = result.get("text", "").strip()

            if self.autocorrect and text and (self.recipe_names or self.ingredients):
                with _timer("Auto-Correct (Fuzzy)", timings):
                    corrected = self.asr_corrector.correct_asr_text_phonetic(
                        text, self.recipe_names, self.ingredients,
                        self.fuzzy_score_cutoff,
                    )
                if corrected != text:
                    print(f"[ASR] Corrected: '{corrected}'")
                text = corrected

            # Persist transcript
            if text:
                out_file = f"recording_{ts_str}.txt"
                self.asr.save_transcription(
                    text, asr_text_dir=self.asr_text_dir, filename=out_file
                )

            return text
        except Exception as e:
            print(f"[ASR] Error: {e}")
            return ""

    @staticmethod
    def _extract_spoken_text(llm_output) -> str:
        """
        If the LLM returned a structured dict, flatten it into a spoken string.
        Otherwise return the raw string.
        """
        if isinstance(llm_output, dict):
            parts = []
            if llm_output.get("greeting"):
                parts.append(llm_output["greeting"])
            if llm_output.get("ingredients"):
                parts.extend(llm_output["ingredients"])
            for step in llm_output.get("steps", []):
                if isinstance(step, dict):
                    parts.append(step.get("text", ""))
                else:
                    parts.append(str(step))
            if llm_output.get("closing"):
                parts.append(llm_output["closing"])
            return " ".join(p for p in parts if p)

        return str(llm_output) if llm_output else ""

    def _shutdown(self) -> None:
        """Graceful cleanup on exit."""
        self.wake_detector.stop()
        with self._player_lock:
            if self._player:
                self._player.stop()
        self._base_player.shutdown()
        print("[Main] Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hey Cook — Voice Recipe Assistant"
    )
    asr_grp = parser.add_mutually_exclusive_group()
    asr_grp.add_argument("--asr-local", action="store_const", const="local",
                         dest="asr_type", help="Use local Whisper for ASR")
    asr_grp.add_argument("--asr-api",   action="store_const", const="api",
                         dest="asr_type", help="Use Deepgram API for ASR (default)")
    parser.add_argument("--model", default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (local ASR only)")
    parser.add_argument("--sensitivity", type=float, default=0.65,
                        help="Wake word detection sensitivity 0–1 (default 0.65)")
    parser.add_argument("--tts", default="deepgram",
                        choices=["deepgram", "sarvam"],
                        help="TTS provider: 'deepgram' (default) or 'sarvam'")
    parser.add_argument("--no-autocorrect", action="store_true",
                        help="Disable fuzzy phonetic ASR auto-correction")
    args = parser.parse_args()

    VoiceAssistant(
        asr_type        = args.asr_type or "api",
        asr_model       = args.model,
        wake_sensitivity = args.sensitivity,
        tts_provider    = args.tts,
        autocorrect     = not args.no_autocorrect,
    ).run()
