import requests
import io
import base64
import wave
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3
from langdetect import detect  # For language detection

# Constants
SARVAM_API_KEY = "c22aa186-2328-46cb-8f87-322e60d5089b"
SARVAM_API_URL_TRANSLATE = "https://api.sarvam.ai/translate"
SARVAM_API_URL_STT = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_API_URL_TTS = "https://api.sarvam.ai/text-to-speech"

# Headers for API requests
headers = {
    "api-subscription-key": SARVAM_API_KEY,
    "Content-Type": "application/json"
}

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Mapping between langdetect codes and Sarvam AI language codes
LANGUAGE_CODE_MAPPING = {
    "en": "en-IN",  # English
    "hi": "hi-IN",  # Hindi
    "bn": "bn-IN",  # Bengali
    "gu": "gu-IN",  # Gujarati
    "kn": "kn-IN",  # Kannada
    "ml": "ml-IN",  # Malayalam
    "mr": "mr-IN",  # Marathi
    "or": "od-IN",  # Odia
    "pa": "pa-IN",  # Punjabi
    "ta": "ta-IN",  # Tamil
    "te": "te-IN",  # Telugu
}

# Function to map detected language codes to Sarvam AI codes
def map_language_code(lang_code):
    """
    Maps langdetect language codes to Sarvam AI language codes.
    """
    return LANGUAGE_CODE_MAPPING.get(lang_code, "en-IN")  # Default to English if not found

# Function to detect language
def detect_language(text):
    """
    Detects the language of the input text.
    """
    try:
        return detect(text)
    except:
        return "en"  # Default to English if detection fails

# Function to translate text
def translate_text(input_text, source_lang, target_lang, mode="classic-colloquial", speaker_gender="Male", numerals_format="international", output_script=None):
    """
    Function to translate text using Sarvam AI API.
    """
    payload = {
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "speaker_gender": speaker_gender,
        "mode": mode,
        "model": "mayura:v1",
        "enable_preprocessing": False,
        "numerals_format": numerals_format,
        "input": input_text
    }
    
    if output_script:
        payload["output_script"] = output_script
    
    response = requests.post(SARVAM_API_URL_TRANSLATE, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("translated_text", "Translation not available")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Function to convert speech to text
def speech_to_text(audio_file_path):
    """
    Function to convert speech to text using Sarvam AI API.
    """
    # Load the audio file
    audio = AudioSegment.from_file(audio_file_path)
    chunk_buffer = io.BytesIO()
    audio.export(chunk_buffer, format="wav")
    chunk_buffer.seek(0)

    files = {'file': ('audiofile.wav', chunk_buffer, 'audio/wav')}
    data = {
        "model": "saaras:v2",
        "with_diarization": False
    }

    try:
        response = requests.post(SARVAM_API_URL_STT, headers=headers, files=files, data=data)
        if response.status_code == 200:
            return response.json().get("transcript", "")
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    finally:
        chunk_buffer.close()

# Function to convert text to speech
def text_to_speech(text, target_lang="hi-IN", speaker="neel"):
    """
    Function to convert text to speech using Sarvam AI API.
    """
    payload = {
        "inputs": [text],
        "target_language_code": target_lang,
        "speaker": speaker,
        "model": "bulbul:v1",
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.0,
        "enable_preprocessing": True
    }

    response = requests.post(SARVAM_API_URL_TTS, json=payload, headers=headers)
    
    if response.status_code == 200:
        audio = base64.b64decode(response.json()["audios"][0])
        with wave.open("output.wav", "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(audio)
        print("Audio file saved as 'output.wav'")
        return "output.wav"
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Function to record real-time audio
def record_audio(duration=5, sample_rate=16000):
    """
    Records audio from the microphone in real-time.
    """
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    write("input.wav", sample_rate, audio)  # Save as WAV file
    return "input.wav"

# Function to play audio
def play_audio(file_path):
    """
    Plays an audio file.
    """
    audio = AudioSegment.from_file(file_path)
    play(audio)

# Main chatbot function
def chatbot():
    print("Welcome to the Multilingual Loan Advisory Chatbot!")
    print("How would you like to interact? (1: Speech, 2: Text)")
    interaction_mode = input("Enter your choice (1 or 2): ")

    while True:
        if interaction_mode == "1":
            # Speech interaction
            print("Please speak now...")
            audio_file_path = record_audio(duration=5)  # Record 5 seconds of audio
            user_input = speech_to_text(audio_file_path)
            print(f"You said: {user_input}")
        elif interaction_mode == "2":
            # Text interaction
            user_input = input("You: ")
        else:
            print("Invalid choice. Exiting.")
            return

        # Detect the input language
        detected_lang = detect_language(user_input)
        input_lang = map_language_code(detected_lang)
        print(f"Detected input language: {detected_lang} -> Mapped to: {input_lang}")

        # Translate user input to English for processing
        if input_lang != "en-IN":
            translated_input = translate_text(user_input, source_lang=input_lang, target_lang="en-IN")
            print(f"Translated input (English): {translated_input}")
        else:
            translated_input = user_input

        # Generate a response (example response)
        response_text = "Your loan application has been processed successfully. Please check your email for further details."
        print(f"Bot response (English): {response_text}")

        # Translate response to the input language
        if input_lang != "en-IN":
            translated_response = translate_text(response_text, source_lang="en-IN", target_lang=input_lang)
            print(f"Translated response: {translated_response}")
        else:
            translated_response = response_text

        # Respond in the same mode as the user
        if interaction_mode == "1":
            # Speech response
            audio_file = text_to_speech(translated_response, target_lang=input_lang)
            if audio_file:
                print("Bot is speaking...")
                play_audio(audio_file)
        elif interaction_mode == "2":
            # Text response
            print(f"Bot: {translated_response}")

        # Ask if the user wants to continue
        print("Do you want to continue? (yes/no)")
        if input().lower() != "yes":
            break

# Run the chatbot
if __name__ == "__main__":
    chatbot()
