import os
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
import wave

class RecipeTTS:
    def __init__(self, output_dir="tts_generated_speech"):
        """
        Initialize the Recipe Text-to-Speech module using Google Gemini TTS API
        
        Args:
            output_dir (str): Directory to save generated speech files
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv("Gemini_API_key")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash-preview-tts"
        
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"TTS module initialized. Audio files will be saved to: {self.output_dir}")

    # def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    #     with wave.open(filename, "wb") as wf:
    #         wf.setnchannels(channels)
    #         wf.setsampwidth(sample_width)
    #         wf.setframerate(rate)
    #         wf.writeframes(pcm)
    
    def generate_speech(self, text, output_filename=None, audio_format="wav"):
        """
        Generate speech from text using Gemini TTS API
        
        Args:
            text (str): The text to convert to speech
            output_filename (str, optional): Custom filename for the output audio. 
                                           If None, uses timestamp
            audio_format (str): Audio format to save as ('wav' or 'mp3'). 
                               Default is 'wav' since Gemini TTS returns WAV format.
            
        Returns:
            str: Path to the saved audio file, or None on error
        """
        try:
            if not text or not text.strip():
                print("Error: Empty text provided for TTS")
                return None
            
            print(f"Generating speech for: {text[:100]}...")

            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"recipe_response_{timestamp}.wav"
            
            
            client = genai.Client(api_key=os.getenv("Gemini_API_key"))
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=f"Say: {text}",
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name='Kore',
                            )
                        )
                    ),
                )
            )

            data = response.candidates[0].content.parts[0].inline_data.data
            
            output_path = os.path.join(self.output_dir, output_filename)

            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(data)
            print(f"✓ Speech generated and saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None
    
    def generate_and_play_speech(self, text, output_filename=None):
        """
        Generate speech and optionally play it using system audio player
        
        Args:
            text (str): The text to convert to speech
            output_filename (str, optional): Custom filename for the output audio
            
        Returns:
            str: Path to the saved audio file, or None on error
        """
        # First generate the speech
        audio_path = self.generate_speech(text, output_filename)
        
        if audio_path:
            # Try to play the audio using system player
            try:
                import subprocess
                # Use common audio players available on Linux
                players = ['mpv', 'ffplay', 'paplay', 'aplay']
                
                for player in players:
                    try:
                        subprocess.Popen([player, audio_path], 
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
                        print(f"Playing audio with {player}...")
                        return audio_path
                    except FileNotFoundError:
                        continue
                
                # If no player found, just return the path
                print(f"Audio file ready at: {audio_path}")
                print("(No audio player found. Please play the file manually.)")
                return audio_path
                
            except Exception as e:
                print(f"Could not play audio: {str(e)}")
                print(f"Audio file saved at: {audio_path}")
                return audio_path
        
        return None
