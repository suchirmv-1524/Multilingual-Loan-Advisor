import openai
import pyttsx3  # For text-to-speech, if you're using a local library like pyttsx3
import os  # For checking file existence

# Initialize OpenAI API
openai.api_key = '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'  # Replace with your actual OpenAI API key

# Speech-to-text using OpenAI's Whisper model
def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file
        )
    return transcript["text"]

# Text-to-speech using pyttsx3 (Local TTS)
def text_to_speech(input_text, language="en_us"):
    # Using pyttsx3 for local speech synthesis
    engine = pyttsx3.init()

    # Set properties (Optional, you can adjust speed, volume, etc.)
    engine.setProperty("rate", 150)  # Speed of speech
    engine.setProperty("volume", 1)  # Volume level (0.0 to 1.0)

    # Set voice based on language preference (you can map language codes to voices)
    voices = engine.getProperty("voices")
    if language == "en_us":
        engine.setProperty("voice", voices[0].id)  # Assuming en_us is the first available voice
    elif language == "hi_in":
        engine.setProperty("voice", voices[1].id)  # Assuming hi_in is the second voice
    else:
        engine.setProperty("voice", voices[0].id)  # Default to en_us

    # Save the speech to a file
    webm_file_path = "temp_audio_play.mp3"
    engine.save_to_file(input_text, webm_file_path)

    # Run the speech engine
    engine.runAndWait()

    return webm_file_path

# Play the audio file automatically using pyttsx3 (local play)
def autoplay_audio(audio_file):
    if os.path.exists(audio_file):
        # Using pyttsx3 to play the audio file directly
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1)

        # Play the audio
        engine.save_to_file(audio_file, "temp_audio_play.mp3")
        engine.runAndWait()
    else:
        print(f"Audio file {audio_file} does not exist.")

# Generate chatbot response using OpenAI's GPT-4
def get_answer(messages):
    system_message = [{"role": "system", "content": "You are a helpful AI chatbot."}]
    messages = system_message + messages
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response.choices[0].message['content']
