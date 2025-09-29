# Cooking Voice Assistant

A voice-controlled cooking assistant that listens for a wake word ("Hey Cook"), records user queries, transcribes speech to text, retrieves relevant recipe information, and responds with cooking instructions using speech synthesis.

## Project Overview

This voice assistant is designed to help with cooking tasks by providing hands-free access to recipes and cooking instructions. The system integrates several components:

1. **Wake Word Detection** - Listens for "Hey Cook" using Picovoice Porcupine
2. **Voice Activity Detection (VAD)** - Records user speech after wake word detection
3. **Automatic Speech Recognition (ASR)** - Transcribes voice recordings using OpenAI Whisper
4. **Recipe Retrieval** - Searches for relevant recipes based on user queries
5. **Text-to-Speech (TTS)** - Converts recipe information to speech responses

## Project Structure

```
BTP_Voice_Assistant/
├── ASR_text/                  # Transcribed voice recordings
├── data/                      # Recipe data files
│   ├── chunks.csv
│   ├── recipes.csv
│   └── searchable_text_for_embeddings.csv
├── data_pipeline/             # Data processing scripts
│   ├── api_call.py            # Recipe API client
│   ├── chunker.py             # Text chunking utilities
│   ├── main.py                # Data pipeline main script
│   └── normalizer.py          # Text normalization utilities
├── models/                    # Model files
│   ├── Food2Vec.bin           # Food embedding model
│   └── Hey-Cook_en_linux_v3_0_0.ppn  # Wake word model
├── modules/                   # Core functionality modules
│   ├── asr.py                 # Automatic Speech Recognition
│   ├── llm.py                 # Language Model integration
│   ├── retriever.py           # Recipe retrieval system
│   ├── tts.py                 # Text-to-Speech synthesis
│   ├── vad.py                 # Voice Activity Detection
│   └── wakeword.py            # Wake word detection
├── src/                       # Application source code
│   └── main.py                # Main application entry point
├── tests/                     # Test scripts
├── voice_recordings/          # Recorded voice input files
└── environment.yaml           # Conda environment specification
```

## Installation

### Prerequisites

- Linux operating system
- Miniconda or Anaconda

### Setup Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd BTP_Voice_Assistant
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate voice-assistant
   ```

3. Add .env file

4. Add the models in the BTP_VOICE_ASSISTANT/models. Link for the models -> https://drive.google.com/drive/folders/1dsdbpFFLysRrCLVCtEo9cCJdg7WuxulV?usp=sharing


## Usage

1. Activate the conda environment:
   ```bash
   conda activate voice-assistant
   ```

2. Run the voice assistant:
   ```bash
   python src/main.py
   ```

3. Say "Hey Cook" to activate the assistant, then speak your query.
   - The assistant will record your speech
   - Transcribe the audio
   - Process your request
   - Respond with relevant recipe information

4. Press `Ctrl+C` to exit the application.

## Data Pipeline

The project includes a data pipeline for processing recipe data:

1. `data_pipeline/api_call.py` - Fetches recipes from external APIs
2. `data_pipeline/chunker.py` - Divides recipes into searchable chunks
3. `data_pipeline/normalizer.py` - Normalizes recipe text, ingredients, and instructions
4. `data_pipeline/main.py` - Orchestrates the data processing workflow

To run the data pipeline:
```bash
python data_pipeline/main.py
```

## Core Components

### Wake Word Detection
Uses Picovoice Porcupine to detect the wake phrase "Hey Cook" with configurable sensitivity.

### Voice Activity Detection (VAD)
Records user speech after wake word detection, with silence detection to automatically stop recording.

### Automatic Speech Recognition (ASR)
Utilizes OpenAI's Whisper model for accurate speech-to-text conversion.

### Recipe Retrieval
Searches through a database of recipes to find relevant information based on user queries.

### Text-to-Speech (TTS)
Converts text responses to speech for hands-free interaction.

## Development

This project is part of a B.Tech Project (BTP) focused on creating an intelligent voice assistant for cooking scenarios.

### Future Enhancements
- Improved recipe understanding and context awareness
- Multi-turn conversations and memory of previous interactions
- Customizable wake word options
- Support for multiple languages
- Integration with smart kitchen devices

## License

[Specify license information]

## Contributors

[List contributors]