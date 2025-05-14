import os
import streamlit as st
from app.stt_elevenlabs import transcribe_audio

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("🎤 Voice Q&A: Upload or Record")

st.markdown(
    "Choose **Upload** to select an audio file, or **Record** to capture from your microphone."
)

# To upload file 
audio_file = st.file_uploader(
    "Upload audio (MP3/WAV):",
    type=["mp3", "wav"]
)  
# To Record
audio_recording = st.audio_input(
    "Or record your question:",
    label_visibility="visible"
)  

if audio_file or audio_recording:
    if audio_file:
        # Uploaded file: already holds bytes
        save_path = os.path.join(UPLOAD_DIR, audio_file.name)
        with open(save_path, "wb") as f:
            f.write(audio_file.getbuffer())
        st.success(f"Uploaded and saved to `{save_path}`")
    else:
        # Recorded via mic: read bytes before writing
        save_path = os.path.join(UPLOAD_DIR, "mic_recording.wav")
        with open(save_path, "wb") as f:
            f.write(audio_recording.read())  # read bytes from BytesIO :contentReference[oaicite:2]{index=2}
        st.success(f"Recorded and saved to `{save_path}`")

        # Playback the recording
        st.audio(audio_recording, format="audio/wav")  # media player :contentReference[oaicite:3]{index=3}

    # Transcribe using ElevenLabs
    with st.spinner("Transcribing audio…"):
        transcript = transcribe_audio(save_path)

    # Display results
    st.markdown("**📝 Transcription:**")
    st.write(transcript)

    # Also log to console
    print("[Converted Text]", transcript) 

    
    # This transcript variable has the text converted from audio, use this further for prompt building with RAG
    
