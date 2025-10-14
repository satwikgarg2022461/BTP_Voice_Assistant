import os
import whisper
import glob
from datetime import datetime
import time
import torch
import csv
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import string
import jellyfish  # For phonetic matching


# Define stopwords to ignore during correction
STOPWORDS = {
    "make", "please", "find", "show", "give", "tell", "cook", "recipe",
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", 
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", 
    "can", "can't", "cannot", "could", "couldn't", 
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", 
    "each", 
    "few", "for", "from", "further", 
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", 
    "hers", "herself", "him", "himself", "his", "how", "how's", 
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", 
    "let's", "it?",
    "make", "me", "more", "most", "mustn't", "my", "myself", "make",
    "no", "nor", "not", "now", 
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shall", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", 
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", 
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", 
    "under", "until", "up", 
    "very", 
    "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "will", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "get", "got", "want", "need", "like", "show", "tell", "give", "recipe", "cook","garam"
}


class WhisperASR:
    def __init__(self, model_size="base"):
        """
        Initialize Whisper ASR with the specified model size.
        
        Args:
            model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        print(f"Loading Whisper {model_size} model...")
        start_time = time.time()
        self.model = whisper.load_model(model_size)
        load_time = time.time() - start_time
        print(f"Whisper {model_size} model loaded in {load_time:.2f} seconds.")
        
        # Initialize sentence transformer model for text correction
        print("Loading sentence transformer model...")
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence transformer model loaded.")
    
    def transcribe_audio(self, audio_file_path):
        """
        Transcribe the given audio file.
        
        Args:
            audio_file_path (str): Path to the audio file to transcribe
            
        Returns:
            dict: Whisper transcription result containing 'text' and other metadata
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
        print(f"Transcribing: {audio_file_path}")
        start_time = time.time()
        # Setting language='en' to force English-only transcription
        result = self.model.transcribe(audio_file_path, language='en')
        transcription_time = time.time() - start_time
        print(f"Transcription completed in {transcription_time:.2f} seconds")
        
        return result
    
    def get_latest_recording(self, recordings_dir="voice_recordings"):
        """
        Get the path to the latest audio recording in the specified directory.
        
        Args:
            recordings_dir (str): Directory containing the recordings
            
        Returns:
            str: Path to the latest recording, or None if no recordings found
        """
        if not os.path.exists(recordings_dir):
            raise FileNotFoundError(f"Recordings directory not found: {recordings_dir}")
            
        recordings = glob.glob(os.path.join(recordings_dir, "*.wav"))
        
        if not recordings:
            return None
            
        latest_recording = max(recordings, key=os.path.getctime)
        return latest_recording
    
    def save_transcription(self, text, asr_text_dir="ASR_text", filename=None):
        """
        Save the transcription to a text file.
        
        Args:
            text (str): The transcribed text to save
            asr_text_dir (str): Directory to save the transcription
            filename (str, optional): Filename for the transcription, uses timestamp if None
            
        Returns:
            str: Path to the saved transcription file
        """
        os.makedirs(asr_text_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcription_{timestamp}.txt"
        
        file_path = os.path.join(asr_text_dir, filename)
        
        with open(file_path, "w") as f:
            f.write(text)
            
        return file_path
    
    def correct_with_embeddings(self, asr_text, recipe_terms, confidence_threshold=0.7):
        """
        Correct ASR text by comparing each word with recipe terms using embeddings.
        
        Args:
            asr_text (str): The ASR transcription text
            recipe_terms (list): List of recipe terms to compare against
            confidence_threshold (float): Threshold for accepting a correction
            
        Returns:
            str: The corrected text
        """
        # Split text into words
        words = asr_text.split()
        corrected_words = []
        
        # Skip correction if no recipe terms
        if not recipe_terms:
            return asr_text
            
        try:
            # Process each word
            for word in words:
                # Embed current word and all recipe terms
                word_emb = self.st_model.encode(word, convert_to_tensor=True)
                term_embs = self.st_model.encode(recipe_terms, convert_to_tensor=True)
                
                # Calculate cosine similarity
                sims = util.cos_sim(word_emb, term_embs)
                best_match_idx = torch.argmax(sims).item()
                best_match = recipe_terms[best_match_idx]
                
                # Apply correction if similarity exceeds threshold
                if sims[0][best_match_idx] > confidence_threshold:
                    corrected_words.append(best_match)
                else:
                    corrected_words.append(word)
                    
            return " ".join(corrected_words)
        except Exception as e:
            print(f"Error during text correction: {str(e)}")
            return asr_text
    
    def correct_asr_text(self, asr_text, recipe_terms, ingredients=None, score_cutoff=70):
        """
        Correct ASR text by comparing each word with recipe terms and ingredients using fuzzy matching.
        
        Args:
            asr_text (str): The ASR transcription text
            recipe_terms (list): List of recipe terms (names) to compare against
            ingredients (list, optional): List of ingredients to compare against
            score_cutoff (int): Minimum similarity score (0-100) for accepting a correction
            
        Returns:
            str: The corrected text
        """
        # Skip correction if no recipe terms
        if not recipe_terms and not ingredients:
            return asr_text
            
        try:
            # Split text into words
            words = asr_text.lower().split()
            corrected_words = []
            
            # Process each word
            for word in words:
                # First try matching against recipphonetic matching + fuzzy (WRatio)e names
                match = process.extractOne(word, recipe_terms, scorer=fuzz.ratio, score_cutoff=score_cutoff)
                # print("reciepe",word, match)
                if match:
                    corrected_words.append(match[0])
                    continue
                
                # Then try matching against ingredients
                if ingredients:
                    match = process.extractOne(word, ingredients, scorer=fuzz.ratio, score_cutoff=score_cutoff)
                    # print("ingredient",word, match)
                    if match:
                        corrected_words.append(match[0])
                        continue
                
                # If no match, keep original
                corrected_words.append(word)
                # print(corrected_words)
            return " ".join(corrected_words)
        except Exception as e:
            print(f"Error during fuzzy text correction: {str(e)}")
            return asr_text
    
    def correct_asr_text_phonetic(self, asr_text, recipe_terms, ingredients=None, score_cutoff=70):
        """
        Correct ASR text using phonetic matching combined with fuzzy string matching.
        Skip stopwords to only correct meaningful recipe-related terms.
        
        Args:
            asr_text (str): The ASR transcription text
            recipe_terms (list): List of recipe terms (names) to compare against
            ingredients (list, optional): List of ingredients to compare against
            score_cutoff (int): Minimum similarity score (0-100) for accepting a correction
            
        Returns:
            str: The corrected text
        """
        # Skip correction if no recipe terms
        if not recipe_terms and not ingredients:
            return asr_text
            
        try:
            # Split text into words and convert to lowercase
            asr_text = asr_text.strip(string.punctuation)
            words = asr_text.lower().split()
            corrected_words = []
            
            # Create phonetic representations of recipe terms and ingredients for faster matching
            recipe_phonetics = []
            if recipe_terms:
                recipe_phonetics = [(term, jellyfish.metaphone(term)) for term in recipe_terms]
            
            ingredient_phonetics = []
            if ingredients:
                ingredient_phonetics = [(ing, jellyfish.metaphone(ing)) for ing in ingredients]
            
            # Process each word
            for word in words:
                # Skip correction for stopwords
                if word.lower() in STOPWORDS:
                    # print(word,"skip")
                    corrected_words.append(word)
                    continue
                    
                word_phonetic = jellyfish.metaphone(word)
                best_match = None
                best_score = 0
                
                # First try matching against recipe terms using phonetic + WRatio
                for term, term_phonetic in recipe_phonetics:
                    # If phonetic codes match or are similar, check with WRatio
                    phonetic_dist = jellyfish.levenshtein_distance(word_phonetic, term_phonetic)
                    if phonetic_dist <= 2:  # Allow some phonetic difference
                        wratio_score = fuzz.WRatio(word, term)
                        if wratio_score > best_score and wratio_score >= score_cutoff:
                            best_score = wratio_score
                            best_match = term
                
                # If no match found in recipes, try ingredients
                if not best_match and ingredients:
                    for ing, ing_phonetic in ingredient_phonetics:
                        phonetic_dist = jellyfish.levenshtein_distance(word_phonetic, ing_phonetic)
                        if phonetic_dist <= 2:
                            wratio_score = fuzz.WRatio(word, ing)
                            if wratio_score > best_score and wratio_score >= score_cutoff:
                                best_score = wratio_score
                                best_match = ing
                
                # Add the best match or original word
                if best_match:
                    corrected_words.append(best_match)
                else:
                    corrected_words.append(word)
                    
            return " ".join(corrected_words)
        except Exception as e:
            print(f"Error during phonetic text correction: {str(e)}")
            return asr_text
    
    def load_recipe_terms(self, csv_path):
        """
        Load recipe names and ingredients from a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file containing recipe data
            
        Returns:
            tuple: (recipe_names, ingredients) as separate lists
        """
        recipe_names = set()
        ingredients = set()
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Add recipe name to terms
                    if row.get('recipe_name'):
                        recipe = row['recipe_name'].lower().split()
                        for i in recipe:
                            clean_word = i.strip(string.punctuation)
                            recipe_names.add(clean_word)
                        # recipe_names.add(row['recipe_name'].lower().split())
                    
                    # Add individual ingredients to terms
                    if row.get('ingredients'):
                        ingredient_list = row['ingredients'].split(', ')
                        ingredient_list_1 =[]
                        for i in ingredient_list:
                            for j in i.split():
                                clean_word = j.strip(string.punctuation)
                                ingredient_list_1.append(clean_word)
                        for ingredient in ingredient_list_1:
                            ingredient = ingredient.strip('"').lower()
                            ingredients.add(ingredient)
            print(recipe_names,"\n\n\n")
            print(ingredients,"\n\n\n")
            
            # Convert sets to lists for matching
            return list(recipe_names), list(ingredients)
            
        except Exception as e:
            print(f"Error loading recipe terms: {str(e)}")
            return [], []