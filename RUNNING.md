# Running the Cooking Voice Assistant

This guide covers end-to-end setup, including creating a **new Picovoice wake-word key** and a **new Upstash Redis URL**.

## 1. Prerequisites

- Linux machine with microphone access
- Conda (Miniconda or Anaconda)
- Python environment from `environment.yaml`

## 2. Clone and set up environment

```bash
git clone <your-repo-url>
cd BTP_Voice_Assistant
conda env create -f environment.yaml
conda activate voice-assistant
```

## 3. Download required model/assets

Make sure these exist before running:

- `models/Hey-Cook_en_linux_v3_0_0.ppn`
- `recipes_demo.db`
- files under `data/`

Model folder reference (project note):  
https://drive.google.com/drive/folders/1dsdbpFFLysRrCLVCtEo9cCJdg7WuxulV?usp=sharing

## 4. Create new Picovoice wake-word access key

Reference: https://picovoice.ai/docs/porcupine/

1. Go to Picovoice Console and sign up/login.
2. Copy your **AccessKey** from the console home page.
3. Put it in `.env` as `PORCUPINE_ACCESS_KEY`.
4. (Optional) Train/download your own `.ppn` wake-word model from the Porcupine console and place it in `models/`.

## 5. Create new Upstash Redis database URL

Reference: https://upstash.com/docs/redis/overall/getstarted

1. Create a new Redis database in Upstash.
2. Open the database **Connect** section.
3. Copy the TLS Redis URL (`rediss://...`), which looks like:
   `rediss://default:<TOKEN>@<HOST>:6379`
4. Put it in `.env` as `REDIS_URL`.

> Note: In this project, Redis is used for session state/persistence (and can support queue-like flows if extended).

## 6. Create `.env`

Create a `.env` file at repository root with:

```bash
# Required for wake-word detection
PORCUPINE_ACCESS_KEY="YOUR_NEW_PICOVOICE_ACCESS_KEY"

# Required for Redis-backed session persistence
REDIS_URL="rediss://default:YOUR_UPSTASH_TOKEN@YOUR_UPSTASH_HOST:6379"

# Required for default runtime path
Gemini_API_key="YOUR_GEMINI_KEY"
Deepgram_API_key="YOUR_DEEPGRAM_KEY"

# Optional if using Sarvam TTS (--tts sarvam)
Sarvam_API_key="YOUR_SARVAM_KEY"

# Optional for API-based full recipe fetch
API_BASE_URL="http://<host>:<port>/recipe2-api"
```

## 7. Run the assistant

```bash
#for web running
python web/app.py

python src/main.py
```

Then say **"Hey Cook"** and speak your command.

### Useful runtime flags

```bash
# Use local Whisper instead of Deepgram ASR
python src/main.py --asr-local --model base

# Use Sarvam TTS
python src/main.py --tts sarvam

# Tune wake-word sensitivity
python src/main.py --sensitivity 0.70
```

## 8. Troubleshooting

- **`Missing PORCUPINE_ACCESS_KEY`**: set valid Picovoice AccessKey in `.env`.
- **`REDIS_URL` warning**: app will run in degraded mode without persistence if Redis is not configured or unreachable.
- **No audio input**: check microphone permissions and default input device.
- **ASR/TTS key errors**: verify `Deepgram_API_key` (or use `--asr-local` plus a working TTS provider).
