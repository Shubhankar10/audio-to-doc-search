# streamlit_app.py

import os
import streamlit as st
from app.stt_elevenlabs import transcribe_audio
from app.tts_elevenlabs import list_voices, text_to_speech
from app.rag_pipeline import llm_response_finance  # or your llm_response function
from app.rag_pipeline import llm_response_sit  # or your llm_response function

# —————————————————————————————
# Setup
# —————————————————————————————

st.set_page_config(page_title="Voice-Driven RAG Q&A", layout="centered")

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state
if "response" not in st.session_state:
    st.session_state.response = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Fetch TTS voices once
voices = list_voices().get("voices", [])
voice_map = {v["name"]: v["voice_id"] for v in voices}

# —————————————————————————————
# Sidebar: Voice Selection
# —————————————————————————————

with st.sidebar:
    st.header("TTS Voice")
    voice_name = st.selectbox(
        "Choose a voice:",
        options=list(voice_map.keys()),
        key="voice_select"
    )
    voice_id = voice_map[voice_name]

# —————————————————————————————
# Main UI
# —————————————————————————————

st.title("🎤 SIT Voice Chatbot")
st.markdown("Send a voice note and get a voice note reply. Continue the conversation naturally!")

# Upload widget
audio_file = st.file_uploader(
    "Send a voice note (MP3/WAV):",
    type=["mp3", "wav"],
    key="upload_file"
)

# Try to use st.audio_input if available (for future compatibility), else fallback to file_uploader only
try:
    audio_recording = st.audio_input(
        "Or record your question:",
        label_visibility="visible",
        key="record_audio"
    )
    audio_input_supported = True
except AttributeError:
    audio_recording = None
    audio_input_supported = False

# Handle new user message
if audio_file or (audio_input_supported and audio_recording):
    if audio_file:
        path = os.path.join(UPLOAD_DIR, audio_file.name)
        with open(path, "wb") as f:
            f.write(audio_file.getbuffer())
        user_audio_bytes = audio_file.getvalue()
    elif audio_input_supported and audio_recording:
        path = os.path.join(UPLOAD_DIR, "mic_recording.wav")
        with open(path, "wb") as f:
            f.write(audio_recording.read())
        user_audio_bytes = audio_recording.read()
    else:
        user_audio_bytes = None

    # Transcribe
    with st.spinner("📝 Transcribing your voice note…"):
        user_text = transcribe_audio(path, language="en")

    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_text,
        "audio": user_audio_bytes
    })

    # Get bot response
    with st.spinner("🤖 Thinking and generating reply…"):
        bot_text = llm_response_sit(user_text)
        bot_audio = text_to_speech(text=bot_text, voice_id=voice_id)

    # Add bot message to chat history
    st.session_state.chat_history.append({
        "role": "bot",
        "text": bot_text,
        "audio": bot_audio
    })

# Display chat history
st.markdown("---")
st.markdown("### Chat History")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown("**You:**")
        st.write(msg["text"])
        if msg["audio"]:
            st.audio(msg["audio"], format="audio/wav")
    else:
        st.markdown("**SIT Bot:**")
        st.write(msg["text"])
        if msg["audio"]:
            st.audio(msg["audio"], format="audio/mp3")
