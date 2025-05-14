# streamlit_app.py

import os
import streamlit as st
from app.stt_elevenlabs import transcribe_audio
from app.tts_elevenlabs import list_voices, text_to_speech

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("🎤 Voice Q&A: Upload, Record, Transcribe & Speak")

# --- Language & Voice Selection ---
# (Optional) fetch voices once
voices = list_voices().get("voices", [])
voice_map = {v["name"]: v["voice_id"] for v in voices}
voice_name = st.selectbox("Choose a voice:", options=list(voice_map.keys()))
voice_id = voice_map[voice_name]

st.markdown("---")

# --- Audio Input Section ---
st.header("1️⃣ Provide Your Question")
st.markdown("Either upload an audio file or record live from your microphone.")

# Upload
audio_file = st.file_uploader("Upload audio (MP3/WAV):", type=["mp3", "wav"])

# Record (if available)
audio_recording = st.audio_input("Or record your question:", label_visibility="visible")

# Process input
if audio_file or audio_recording:
    if audio_file:
        save_path = os.path.join(UPLOAD_DIR, audio_file.name)
        with open(save_path, "wb") as f:
            f.write(audio_file.getbuffer())
        st.success(f"Uploaded and saved to `{save_path}`")
    else:
        save_path = os.path.join(UPLOAD_DIR, "mic_recording.wav")
        # audio_recording is an UploadedFile-like object
        with open(save_path, "wb") as f:
            f.write(audio_recording.read())
        st.success(f"Recorded and saved to `{save_path}`")
        st.audio(audio_recording, format="audio/wav")

    # --- Transcription ---
    st.header("2️⃣ Transcription")
    with st.spinner("Transcribing audio…"):
        transcript = transcribe_audio(save_path, language="en")
    st.markdown("**📝 Transcript:**")
    st.write(transcript)
    print("[Transcription]", transcript)





    # --- Text-to-Speech ---
    st.header("3️⃣ Listen to the Answer")
    if st.button("🔉 Play Transcript"):
        with st.spinner("Generating speech…"):
            audio_bytes = text_to_speech(text=transcript, voice_id=voice_id)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.error("TTS generation failed. Check console for details.")
