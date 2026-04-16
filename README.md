# Cooking Voice Assistant

Voice-first recipe assistant that listens for **"Hey Cook"**, records speech, detects intent, and guides users through recipes using spoken responses.

## Quick Start

1. Create and activate the environment:
   ```bash
   conda env create -f environment.yaml
   conda activate voice-assistant
   ```
2. Configure your `.env` (required keys include `PORCUPINE_ACCESS_KEY` and `REDIS_URL`; see details below).
3. Ensure required assets exist:
   - `models/Hey-Cook_en_linux_v3_0_0.ppn`
   - `recipes_demo.db`
   - data files under `data/`
4. Run:
   ```bash
   python src/main.py
   ```

For full setup (including how to create new Picovoice and Upstash credentials), see **[RUNNING.md](./RUNNING.md)**.

## Required Environment Variables

These names are case-sensitive and match the current code:

```bash
PORCUPINE_ACCESS_KEY="..."                   # Picovoice Porcupine wake-word auth
REDIS_URL="rediss://default:<TOKEN>@<HOST>:6379"  # Upstash Redis URL
Gemini_API_key="..."                         # Gemini (intent + LLM)
Deepgram_API_key="..."                       # Deepgram (ASR + default TTS)
API_BASE_URL="http://<host>:<port>/recipe2-api"   # Optional API for full recipe fetch
# Optional (needed only if using --tts sarvam)
Sarvam_API_key="..."
```

## Key Links

- Picovoice Porcupine: https://picovoice.ai/docs/porcupine/
- Upstash Redis getting started: https://upstash.com/docs/redis/overall/getstarted

## Project Layout

```text
BTP_Voice_Assistant/
├── src/main.py                 # Voice assistant entry point
├── modules/                    # Wakeword, ASR, intent, retrieval, session, TTS
├── models/                     # Wake-word .ppn and model artifacts
├── data/                       # Recipe CSV data
├── data_pipeline/              # Data processing scripts
├── recipes_demo.db             # Vector/search database
└── environment.yaml            # Conda environment definition
```

## Common Run Modes

```bash
#for web running
python web/app.py

# Default: Deepgram ASR + Deepgram TTS
python src/main.py

# Local Whisper ASR
python src/main.py --asr-local --model base

# Sarvam TTS
python src/main.py --tts sarvam


```
