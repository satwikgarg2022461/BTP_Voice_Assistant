"""
Web interface for the BTP Voice Assistant.

Thin Flask wrapper around the existing pipeline modules.
Wake word and hardware VAD are replaced by browser-side recording;
the uploaded WAV goes straight to ASR.
"""

import os
import sys
import threading
import tempfile
from datetime import datetime
from typing import Optional

from flask import Flask, request, jsonify, send_from_directory, send_file

# Project root on sys.path so `modules.*` imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.deepgram_asr     import DeepgramASR
from modules.asr              import WhisperASR
from modules.intent_classifier import IntentClassifier, Intent
from modules.navigator         import RecipeNavigator
from modules.session_manager   import SessionManager
from modules.retriever         import RecipeRetriever
from modules.llm              import RecipeLLM
from modules.tts              import get_tts_engine

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static", static_url_path="/static")

TTS_DIR     = os.path.join(PROJECT_ROOT, "tts_generated_speech")
UPLOAD_DIR  = os.path.join(PROJECT_ROOT, "voice_recordings")
ASR_DIR     = os.path.join(PROJECT_ROOT, "ASR_text")
SESSION_ID  = "web_user_001"

os.makedirs(TTS_DIR,    exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ASR_DIR,    exist_ok=True)


class WebAssistant:
    """Holds all heavy module instances (created once at startup)."""

    def __init__(self, asr_type="local", asr_model="small",
                 tts_provider="sarvam", autocorrect=False):
        print("=" * 60)
        print("  Initialising Web Voice Assistant")
        print(f"  ASR: {asr_type}  |  TTS: {tts_provider}  |  autocorrect: {autocorrect}")
        print("=" * 60)

        # ASR
        if asr_type == "api":
            self.asr = DeepgramASR()
            print("ASR  : Deepgram API")
        else:
            self.asr = WhisperASR(model_size=asr_model)
            print(f"ASR  : Whisper local ({asr_model})")

        self.autocorrect = autocorrect

        self.classifier = IntentClassifier(confidence_threshold=0.70,
                                           use_llm_fallback=True)
        self.navigator  = RecipeNavigator()
        self.session_mgr = SessionManager(session_ttl=3600)
        self.retriever  = RecipeRetriever(
            db_path=os.path.join(PROJECT_ROOT, "recipes_demo.db"),
            collection_name="recipes_collection",
        )

        try:
            self.llm = RecipeLLM()
            print("LLM  : OK")
        except Exception as e:
            print(f"LLM  : unavailable ({e})")
            self.llm = None

        try:
            self.tts = get_tts_engine(tts_provider, output_dir=TTS_DIR)
            print(f"TTS  : OK ({tts_provider})")
        except Exception as e:
            print(f"TTS  : unavailable ({e})")
            self.tts = None

        self.fuzzy_score_cutoff = 70
        self.recipe_names: list = []
        self.ingredients:  list = []
        if autocorrect:
            food_dict = os.path.join(PROJECT_ROOT, "data", "food_dictionary.csv")
            if os.path.exists(food_dict):
                asr_corrector = WhisperASR(model_size="base") if asr_type == "api" else self.asr
                self.recipe_names, self.ingredients = \
                    asr_corrector.load_recipe_terms(food_dict)
                print(f"Vocab: {len(self.recipe_names)} recipes, "
                      f"{len(self.ingredients)} ingredients loaded")
        else:
            print("Vocab: autocorrect disabled")

        self._session: Optional[dict] = None
        self._lock = threading.Lock()

        print("\nWeb Voice Assistant ready.\n")

    # ---------- transcription ----------

    def transcribe(self, audio_path: str) -> str:
        result = self.asr.transcribe_audio(audio_path)
        text = result.get("text", "").strip()
        if self.autocorrect and text and (self.recipe_names or self.ingredients):
            corrected = self.asr.correct_asr_text_phonetic(
                text, self.recipe_names, self.ingredients,
                self.fuzzy_score_cutoff,
            )
            if corrected != text:
                print(f"[ASR] Corrected: '{corrected}'")
            text = corrected
        return text

    # ---------- intent dispatch ----------

    def process(self, text: str) -> dict:
        """Run intent classification + dispatch, return a response dict."""
        context = {}
        with self._lock:
            if self._session:
                context = {
                    "current_state": self._session.get("current_state", ""),
                    "recipe_title":  self._session.get("recipe_title", ""),
                }

        intent, confidence, entities = self.classifier.classify(text, context)

        response_text = ""
        sidebar = self._sidebar_state()

        if intent in (Intent.NAV_NEXT, Intent.NAV_PREV, Intent.NAV_GO_TO,
                      Intent.NAV_REPEAT, Intent.NAV_REPEAT_INGREDIENTS,
                      Intent.NAV_START):
            response_text = self._handle_navigation(intent, entities)

        elif intent == Intent.SEARCH_RECIPE:
            response_text = self._handle_recipe_search(text)

        elif intent == Intent.START_RECIPE:
            response_text = self._handle_start_recipe()

        elif intent == Intent.QUESTION:
            response_text = self._handle_question(text)

        elif intent == Intent.SMALL_TALK:
            response_text = self._handle_small_talk(text)

        elif intent == Intent.STOP_PAUSE:
            response_text = "Okay, I've paused. Say something when you're ready."

        elif intent == Intent.RESUME:
            response_text = "Resuming! What would you like to do next?"

        elif intent == Intent.HELP:
            response_text = (
                "You can say: next step, previous step, repeat, "
                "repeat ingredients, go to step 3, restart, pause, "
                "resume, or ask me any question about the recipe."
            )

        elif intent == Intent.CONFIRM:
            response_text = self._handle_confirm()

        elif intent == Intent.CANCEL:
            response_text = "No problem. Just let me know when you're ready."

        else:
            response_text = "I'm not sure what you mean. Try asking about a recipe or saying 'next step'."

        sidebar = self._sidebar_state()

        return {
            "intent":     intent.value,
            "confidence": round(confidence, 2),
            "entities":   entities,
            "text":       response_text,
            "sidebar":    sidebar,
        }

    # ---------- handlers ----------

    def _handle_navigation(self, intent: Intent, entities: dict) -> str:
        with self._lock:
            session = self._session
        if session is None:
            return "No recipe is active. Please search for a recipe first."

        nav = self.navigator
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
            return "I couldn't navigate the recipe."

        with self._lock:
            nav.update_session_from_result(self._session, result)
            sid = self._session.get("session_id", SESSION_ID)

        self.session_mgr.update_session(sid, {
            "step_index":      result.step_index,
            "chunk_index":     result.chunk_index,
            "current_section": result.section,
            "last_intent":     result.intent,
        })

        return result.text

    def _handle_recipe_search(self, text: str) -> str:
        results = self.retriever.search_recipes(text, limit=1)
        if not results:
            return "I couldn't find a matching recipe. Please try again."

        top = results[0]
        recipe_id = top.get("recipe_id")
        recipe_data = self.navigator.load_recipe(recipe_id)
        if recipe_data is None:
            return f"I found a recipe but couldn't load its steps."

        nav_fields = self.navigator.build_session_nav_fields(recipe_data)
        session = self.session_mgr.create_session(
            session_id=SESSION_ID,
            recipe_id=str(recipe_id),
            recipe_title=recipe_data.title,
            total_steps=recipe_data.total_steps,
        )
        session.update(nav_fields)
        session["session_id"] = SESSION_ID

        with self._lock:
            self._session = session

        return (
            f"I found {recipe_data.title}! "
            f"It has {recipe_data.total_steps} steps. "
            f"Say 'start cooking' to begin, or ask me anything."
        )

    def _handle_start_recipe(self) -> str:
        with self._lock:
            session = self._session
        if session is None:
            return "No recipe loaded. Please search for a recipe first."

        ingr_result = self.navigator.get_current_ingredients(session)
        with self._lock:
            self.navigator.update_session_from_result(self._session, ingr_result)

        step_result = self.navigator.get_next_step(self._session)
        with self._lock:
            self.navigator.update_session_from_result(self._session, step_result)

        prompt = "Say 'next step' whenever you're ready to continue."
        return ingr_result.text + " " + step_result.text + " " + prompt

    def _handle_question(self, text: str) -> str:
        if self.llm is None:
            return "I can't answer questions right now."

        with self._lock:
            history = (self._session or {}).get("conversation_history", [])

        results = self.retriever.search_recipes(text, limit=1)
        recipe_context = ""
        if results:
            recipe_context = results[0].get("text_preview", results[0].get("title", ""))

        answer = self.llm.answer_recipe_question(
            text, recipe_context=recipe_context,
            conversation_history=history,
        )
        return self._extract_spoken(answer)

    def _handle_small_talk(self, text: str) -> str:
        if self.llm is None:
            return "I'm your cooking assistant! Let me know when you'd like to start a recipe."
        with self._lock:
            history = (self._session or {}).get("conversation_history", [])
            context = {
                "recipe_title":    (self._session or {}).get("recipe_title", ""),
                "step_index":      (self._session or {}).get("step_index", 0),
                "current_section": (self._session or {}).get("current_section", ""),
            }
        answer = self.llm.generate_conversational_response(
            text, "small_talk",
            conversation_history=history, context=context,
        )
        return answer if isinstance(answer, str) else self._extract_spoken(answer)

    def _handle_confirm(self) -> str:
        with self._lock:
            session = self._session
        if session and session.get("current_state") in ("READING_STEPS", "READING_INGREDIENTS"):
            return self._handle_navigation(Intent.NAV_NEXT, {})
        return "Okay!"

    # ---------- helpers ----------

    @staticmethod
    def _extract_spoken(llm_output) -> str:
        if isinstance(llm_output, dict):
            parts = []
            if llm_output.get("greeting"):
                parts.append(llm_output["greeting"])
            if llm_output.get("ingredients"):
                parts.extend(llm_output["ingredients"])
            for step in llm_output.get("steps", []):
                parts.append(step.get("text", "") if isinstance(step, dict) else str(step))
            if llm_output.get("closing"):
                parts.append(llm_output["closing"])
            return " ".join(p for p in parts if p)
        return str(llm_output) if llm_output else ""

    def _sidebar_state(self) -> dict:
        with self._lock:
            s = self._session or {}
        return {
            "recipe_title":    s.get("recipe_title", ""),
            "current_step":    s.get("step_index", 0),
            "total_steps":     s.get("total_steps", 0),
            "current_section": s.get("current_section", ""),
        }

    def generate_audio(self, text: str) -> Optional[str]:
        if not self.tts or not text.strip():
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunks = self.tts.generate_speech_chunks(text, output_prefix=f"web_{ts}")
        valid = [c for c in chunks if c.get("audio_path") and not c.get("error")]
        if not valid:
            return None
        if len(valid) == 1:
            return os.path.basename(valid[0]["audio_path"])
        return self._concat_audio(valid, f"web_{ts}_full")

    def _concat_audio(self, chunks: list, out_base: str) -> str:
        """Concatenate multiple audio chunk files into one.

        Handles both WAV (Sarvam) and MP3 (Deepgram) correctly:
        - WAV: reads PCM data from each file, writes a single WAV with
          a correct header covering all the combined PCM data.
        - MP3: raw byte concatenation (MP3 frames are self-describing).
        """
        import wave
        import struct

        first_path = chunks[0]["audio_path"]
        is_wav = first_path.lower().endswith(".wav")

        if is_wav:
            out_name = out_base + ".wav"
            out_path = os.path.join(TTS_DIR, out_name)
            params_set = False
            with wave.open(out_path, "wb") as wout:
                for c in chunks:
                    try:
                        with wave.open(c["audio_path"], "rb") as win:
                            if not params_set:
                                wout.setparams(win.getparams())
                                params_set = True
                            wout.writeframes(win.readframes(win.getnframes()))
                    except Exception as e:
                        print(f"[Audio concat] Skipping chunk: {e}")
            return out_name
        else:
            out_name = out_base + ".mp3"
            out_path = os.path.join(TTS_DIR, out_name)
            with open(out_path, "wb") as out:
                for c in chunks:
                    with open(c["audio_path"], "rb") as f:
                        out.write(f.read())
            return out_name


# ---------------------------------------------------------------------------
# Singleton assistant (created lazily or at __main__)
# ---------------------------------------------------------------------------

assistant: Optional[WebAssistant] = None


def get_assistant() -> WebAssistant:
    global assistant
    if assistant is None:
        assistant = WebAssistant()
    return assistant


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/process", methods=["POST"])
def api_process():
    """Accept an audio file, run the full pipeline, return JSON + audio URL."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(UPLOAD_DIR, f"web_recording_{ts}.wav")
    audio_file.save(save_path)

    a = get_assistant()

    # 1. Transcribe
    transcript = a.transcribe(save_path)
    if not transcript:
        return jsonify({
            "transcript": "",
            "response":   "I didn't catch that. Could you try again?",
            "intent":     "unknown",
            "audio_url":  None,
            "sidebar":    a._sidebar_state(),
        })

    # 2. Process (classify + dispatch)
    result = a.process(transcript)

    # 3. Generate TTS audio
    audio_filename = a.generate_audio(result["text"])
    audio_url = f"/api/audio/{audio_filename}" if audio_filename else None

    return jsonify({
        "transcript": transcript,
        "response":   result["text"],
        "intent":     result["intent"],
        "confidence": result["confidence"],
        "audio_url":  audio_url,
        "sidebar":    result["sidebar"],
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Clear session state (in-memory + Redis)."""
    a = get_assistant()
    with a._lock:
        sid = (a._session or {}).get("session_id", SESSION_ID)
        a._session = None
    try:
        a.session_mgr.delete_session(sid)
    except Exception:
        pass
    return jsonify({"ok": True})


@app.route("/api/audio/<path:filename>")
def api_audio(filename):
    """Serve a generated TTS audio file."""
    return send_from_directory(TTS_DIR, filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hey Cook — Web Voice Assistant")
    asr_grp = parser.add_mutually_exclusive_group()
    asr_grp.add_argument("--asr-local", action="store_const", const="local",
                         dest="asr_type", help="Use local Whisper for ASR (default)")
    asr_grp.add_argument("--asr-api",   action="store_const", const="api",
                         dest="asr_type", help="Use Deepgram API for ASR")
    parser.add_argument("--model", default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: small)")
    parser.add_argument("--tts", default="sarvam",
                        choices=["deepgram", "sarvam"],
                        help="TTS provider (default: sarvam)")
    parser.add_argument("--no-autocorrect", action="store_true",
                        help="Disable fuzzy phonetic ASR auto-correction (default: disabled)")
    parser.add_argument("--port", type=int, default=5001,
                        help="Port to run on (default: 5001)")
    args = parser.parse_args()

    assistant = WebAssistant(
        asr_type     = args.asr_type or "local",
        asr_model    = args.model,
        tts_provider = args.tts,
        autocorrect  = args.no_autocorrect,
    )

    app.run(host="0.0.0.0", port=args.port, debug=False)
