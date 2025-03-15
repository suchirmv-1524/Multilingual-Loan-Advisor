import openai
import os
from dotenv import load_dotenv
from langdetect import detect
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from utils import speech_to_text, text_to_speech, get_answer, autoplay_audio

# Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def detect_language(text):
    """Detect the language of the input text."""
    try:
        language = detect(text)
        return language
    except:
        return "en"  # Default to English if detection fails

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How may I assist you today?"}]

initialize_session_state()

st.title("Loan Advisor 🤖")

# Language Selection
language_options = {"English": "en", "Hindi": "hi", "Kannada": "kn", "Telugu": "te", "Tamil": "ta"}
selected_language = st.selectbox("Select Language", list(language_options.keys()))
st.session_state.selected_language = language_options[selected_language]

# User Input (Text)
user_input = st.text_input("Type your message here:")

# User Input (Speech)
audio_bytes = audio_recorder()
if audio_bytes:
    with st.spinner("Transcribing..."):
        audio_path = "temp_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        user_input = speech_to_text(audio_path)
        os.remove(audio_path)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

# Generate Response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking🤔..."):
            final_response = get_answer(st.session_state.messages)
        
        with st.spinner("Generating audio response..."):
            audio_file = text_to_speech(final_response, st.session_state.selected_language)
            autoplay_audio(audio_file)
            os.remove(audio_file)
        
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})