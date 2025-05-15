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

st.title("🎤 Voice-Driven Document Q&A")

st.markdown("### 1️⃣ Provide Your Question")
st.markdown("Upload an audio file or record via your microphone.")

# Upload widget
audio_file = st.file_uploader(
    "Upload audio (MP3/WAV):",
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

# When we have audio, save + transcribe once
if audio_file or (audio_input_supported and audio_recording):
    if audio_file:
        path = os.path.join(UPLOAD_DIR, audio_file.name)
        with open(path, "wb") as f:
            f.write(audio_file.getbuffer())
        st.success(f"📥 Saved upload to `{path}`")
    elif audio_input_supported and audio_recording:
        path = os.path.join(UPLOAD_DIR, "mic_recording.wav")
        with open(path, "wb") as f:
            f.write(audio_recording.read())
        st.success(f"🎤 Saved recording to `{path}`")
        st.audio(audio_recording, format="audio/wav")

    # Transcribe
    with st.spinner("📝 Transcribing…"):
        st.session_state.transcript = transcribe_audio(path, language="en")

# Show editable transcript if available
if st.session_state.transcript:
    st.markdown("### 2️⃣ Transcription (editable)")
    editable = st.text_area(
        "Edit your question before querying the LLM:",
        value=st.session_state.transcript,
        height=150,
        key="editable_transcript"
    )

    # Query LLM
    if st.button("💡 Get Answer", key="get_answer"):
        with st.spinner("🤖 Thinking…"):
            # st.session_state.response = llm_response_finance(editable)
            st.session_state.response = llm_response_sit(editable)

# Display LLM response if we have one
if st.session_state.response:
    st.markdown("### ✅ Response")
    st.write(st.session_state.response)

    # TTS playback
    st.markdown("### 3️⃣ Listen to the Answer")
    if st.button("🔉 Play Answer", key="play_tts"):
        with st.spinner("🔊 Generating speech…"):
            audio_bytes = text_to_speech(
                text=st.session_state.response,
                voice_id=voice_id
            )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.error("TTS generation failed. Check console for details.")
