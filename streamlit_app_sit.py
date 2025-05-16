# streamlit_app.py

import os
import streamlit as st
from app.stt_elevenlabs import transcribe_audio
from app.tts_elevenlabs import list_voices, text_to_speech
from app.rag_pipeline import llm_response_sit

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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Fetch TTS voices once
voices = list_voices().get("voices", [])
voice_map = {v["name"]: v["voice_id"] for v in voices}
default_voice_name = list(voice_map.keys())[0]
voice_id = voice_map[default_voice_name]

# —————————————————————————————
# Main UI
# —————————————————————————————

# SIT Logo
st.image("sit-data/logo.png", width=150)

st.title("🎤 SIT Voice QA Chatbot")
st.markdown("Send a voice note and get a voice note reply. Continue the conversation naturally!")

# Upload widget
audio_file = st.file_uploader(
    "Send a voice note (MP3/WAV):",
    type=["mp3", "wav"],
    key="upload_file"
)

# Optional: Use st.audio_input if available
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

# Handle new message
if (audio_file or (audio_input_supported and audio_recording)) and not st.session_state.get("chat_ended", False):
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

    # Append user message
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_text,
        "audio": user_audio_bytes
    })

    # Get bot response
    with st.spinner("🤖 Thinking and generating reply…"):
        bot_text = llm_response_sit(user_text)
        bot_audio = text_to_speech(text=bot_text, voice_id=voice_id)

    # Append bot message
    st.session_state.chat_history.append({
        "role": "bot",
        "text": bot_text,
        "audio": bot_audio
    })

# Display chat history as a chatbot interface
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 80px;
}
.chat-row {
    display: flex;
    align-items: flex-end;
}
.bot-message {
    background: #f1f0f0;
    color: #222;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    max-width: 60%;
    margin-right: auto;
    box-shadow: 0 2px 8px #0001;
    display: flex;
    align-items: flex-end;
    gap: 8px;
}
.user-message {
    background: #4b8bff;
    color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    max-width: 60%;
    margin-left: auto;
    box-shadow: 0 2px 8px #0001;
    display: flex;
    align-items: flex-end;
    gap: 8px;
    flex-direction: row-reverse;
}
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: cover;
    background: #eee;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}
</style>
<div class="chat-container">
""", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-row">
            <div class="user-message">
                <div class="avatar">🧑</div>
                <div>
                    <div style='margin-bottom:4px;'><b>You</b></div>
                    <div>{msg['text']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if msg["audio"]:
            st.audio(msg["audio"], format="audio/wav")
    else:
        st.markdown(f"""
        <div class="chat-row">
            <div class="bot-message">
                <div class="avatar"><img src='https://img.freepik.com/free-vector/chatbot-chat-message-vectorart_78370-4104.jpg?semt=ais_hybrid&w=740' width='32' height='32'/></div>
                <div>
                    <div style='margin-bottom:4px;'><b>SIT Bot</b></div>
                    <div>{msg['text']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if msg["audio"]:
            st.audio(msg["audio"], format="audio/mp3")

st.markdown("</div>", unsafe_allow_html=True)

# —————————————————————————————
# End Chat Button (bottom right)
# —————————————————————————————
st.markdown(
    """
    <div style='position: fixed; bottom: 20px; right: 20px;'>
        <form action="" method="post">
            <button type="submit" name="clear_chat" style="
                background-color: #ff4b4b;
                border: none;
                color: white;
                padding: 10px 16px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                border-radius: 6px;
                cursor: pointer;
            ">🛑 End Chat & Clear History</button>
        </form>
    </div>
    """,
    unsafe_allow_html=True,
)

# Clear logic
if st.session_state.get("clear_chat", False) or st.query_params.get("clear_chat"):
    st.session_state.chat_history = []
    st.session_state.response = None
    st.session_state.transcript = None
    st.rerun()
