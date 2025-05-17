# streamlit_app_medical_modular.py

import os
import time
import threading

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from app.stt_elevenlabs import transcribe_audio
from app.tts_elevenlabs import list_voices, text_to_speech
from app.rag_pipeline import llm_response_medical_debate
from app.utils import (
    get_custom_css,
    autoplay_audio,
    render_listening_animation,
    render_message_bubbles,
)
from app.vad import VoiceActivityDetector

# —————————————————————————————
# Constants
# —————————————————————————————
UPLOAD_DIR = "uploads"


# —————————————————————————————
# Helper Functions
# —————————————————————————————
def ensure_upload_dir():
    """Ensure the upload directory exists."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def on_silence_detected():
    """
    Callback invoked by VAD when silence is detected.
    Simulates a click on the Record button to stop recording.
    """
    if st.session_state.recording:
        js = """
        <script>
        document.querySelector('button[aria-label="Record"]').click();
        </script>
        """
        components.html(js, height=0)
        st.session_state.recording = False


def process_audio_for_vad(audio_bytes: bytes, rate: int = 44100) -> bool:
    """
    Run Voice Activity Detection on raw audio bytes.

    Args:
        audio_bytes: Raw audio buffer.
        rate: Sample rate (Hz).

    Returns:
        Whether speech is currently active.
    """
    vad = VoiceActivityDetector(
        silence_threshold=0.03,
        silence_duration=st.session_state.vad_timeout,
        on_silence_callback=on_silence_detected,
    )

    # Convert to numpy array if needed
    if not isinstance(audio_bytes, np.ndarray):
        try:
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        except Exception:
            return False
    else:
        audio_data = audio_bytes

    return vad.process_audio(audio_data)


def load_voices():
    """
    Fetch available TTS voices from ElevenLabs and store in session_state.
    """
    try:
        voices = list_voices().get("voices", [])
        voice_map = {v["name"]: v["voice_id"] for v in voices}

        # Set a default voice on first load
        if not st.session_state.get("voice_id"):
            default_name = next(iter(voice_map))
            st.session_state.voice_name = default_name
            st.session_state.voice_id = voice_map[default_name]

        return voice_map

    except Exception as e:
        st.error(f"Error loading voices: {e}")
        # Fallback defaults
        st.session_state.voice_name = "Default"
        st.session_state.voice_id = "default"
        return {"Default": "default"}


# —————————————————————————————
# Session State Initialization
# —————————————————————————————
def init_session_state():
    """
    Ensure all required session_state keys exist with sensible defaults.
    """
    defaults = {
        "response": None,
        "transcript": None,
        "chat_history": [],
        "debate_topic": "",
        "debate_side": "against",
        "debate_started": False,
        "listening": False,
        "recording": False,
        "last_user_input": None,
        "auto_listen": True,
        "vad_timeout": 2.0,
        "audio_buffer": None,
        "should_stop_recording": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# —————————————————————————————
# UI Setup
# —————————————————————————————
def setup_page():
    """Configure the Streamlit page and inject custom CSS & JS."""
    st.set_page_config(
        page_title="Medical Voice Debate",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    # Custom styling
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Keyboard shortcuts & auto-scroll JS
    keyboard_js = """
    <script>
    // Space: toggle record, Esc: stop, F1: help
    document.addEventListener('keydown', e => {
      if (!e.target.matches('input, textarea')) {
        if (e.code==='Space'&&!e.repeat){e.preventDefault();document.querySelector('button[aria-label="Record"]').click();}
        if (e.code==='Escape'){document.querySelector('button[aria-label="Record"].recording')?.click();}
        if (e.code==='F1'){e.preventDefault();document.querySelector('details[data-testid="stExpander"]').open^=1;}
      }
    });
    // Observe record button to sync recording state
    (function observe(){
      const btn=document.querySelector('button[aria-label="Record"]');
      if(btn){
        new MutationObserver(muts=>{
          muts.forEach(m=>{
            const role = m.target.getAttribute('aria-label');
            const rec = role==='Stop';
            m.target.classList.toggle('recording', rec);
            window.parent.postMessage({type:'streamlit:setComponentValue',value:rec},'*');
          });
        }).observe(btn,{attributes:true});
      } else setTimeout(observe,1000);
    })();
    // Auto-scroll chat
    (function scroll(){
      const c=document.getElementById('chat-container');
      if(c){c.scrollTop=c.scrollHeight;}
      setTimeout(scroll,800);
    })();
    </script>
    """
    components.html(keyboard_js, height=0)


# —————————————————————————————
# Debate Setup UI
# —————————————————————————————
def render_setup_panel(voice_map):
    """
    Render the initial debate setup panel:
    - Topic input
    - AI side selection
    - Voice and advanced settings
    - Start Debate button
    """
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("### Start a New Debate")

    # Topic and side
    topic = st.text_input(
        "Enter a medical debate topic:",
        placeholder="e.g., 'AI will replace doctors in the next decade'",
    )
    col1, col2 = st.columns(2)
    with col1:
        side = st.radio("AI should argue:", ["FOR", "AGAINST"], horizontal=True)
    with col2:
        selected = st.selectbox("AI voice:", list(voice_map), index=list(voice_map).index(st.session_state.voice_name))
        if selected != st.session_state.voice_name:
            st.session_state.voice_name, st.session_state.voice_id = selected, voice_map[selected]

    # Advanced settings
    with st.expander("Advanced Settings"):
        st.session_state.auto_listen = st.checkbox(
            "Auto-start listening after AI responds",
            value=st.session_state.auto_listen,
        )
        timeout = st.slider(
            "Seconds of silence before auto-stopping recording",
            1.0, 5.0, st.session_state.vad_timeout, 0.5,
        )
        st.session_state.vad_timeout = timeout

    # Start button
    if st.button("Start Debate", type="primary", use_container_width=True):
        if topic:
            st.session_state.debate_topic = topic
            st.session_state.debate_side = side.lower()
            st.session_state.debate_started = True
            st.session_state.debate_round = 1
            st.session_state.chat_history = []
            st.rerun()
        else:
            st.error("Please enter a debate topic to begin.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Tips
    with st.expander("Tips for using the Voice Debate Assistant"):
        st.markdown(
            """
            - Press the **Space Bar** to start/stop recording  
            - The app auto-listens after each AI response  
            - Customize AI voice in settings  
            - Speak clearly for best transcription  
            - Visual indicators show listening status  
            """
        )


# —————————————————————————————
# Debate Interface UI
# —————————————————————————————
def render_debate_interface():
    """
    Render the ongoing debate interface:
    - Title and topic header
    - Chat history container with user/AI bubbles
    - Voice controls and recording logic
    """
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
          <div><h3 style="margin:0;">{st.session_state.debate_topic}</h3>
          <p style="margin:0;color:#666;">AI argues <strong>{st.session_state.debate_side.upper()}</strong></p></div>
          <div><span class="keyboard-shortcut">Press Space to speak</span></div>
        </div><hr>
        """,
        unsafe_allow_html=True,
    )

    # Show chat history
    with st.container():
        st.markdown(
            "<div id='chat-container' style='height:350px;overflow-y:auto;margin-bottom:20px;padding-right:10px;'>",
            unsafe_allow_html=True,
        )
        for idx, msg in enumerate(st.session_state.chat_history):
            render_message_bubbles(msg, idx)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Listening indicator
    if st.session_state.listening:
        render_listening_animation()
    else:
        st.markdown(
            "<div style='text-align:center;color:#666;margin:15px 0;'>AI is thinking...</div>",
            unsafe_allow_html=True,
        )

    # Controls: New Debate, Record, Replay
    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        if st.button("⏹️ New Debate", type="secondary"):
            st.session_state.debate_started = False
            st.session_state.chat_history = []
            st.rerun()

    with col2:
        audio_input_supported = True
        try:
            audio_data = st.audio_input(
                "Speak your argument...",
                key="voice_input",
                label_visibility="collapsed",
                on_change=lambda: st.session_state.__setitem__("recording", not st.session_state.recording),
            )
            st.markdown(
                "<div style='text-align:center;font-size:0.8em;color:#666;'>"
                "Press Space to speak; stops after you pause"
                "</div>",
                unsafe_allow_html=True,
            )
        except AttributeError:
            audio_input_supported = False
            st.warning("Live voice input not supported. Please upload a recording.")
            audio_data = st.file_uploader("Upload recording (MP3/WAV):", type=["mp3", "wav"])

    with col3:
        if st.session_state.chat_history:
            if st.button("🔄 Replay Last", type="secondary"):
                for m in reversed(st.session_state.chat_history):
                    if m["role"] == "bot" and m["audio"]:
                        st.audio(m["audio"], format="audio/mp3", autoplay=True)
                        break

    # Handle incoming audio (live or upload)
    handle_audio_input(audio_data, audio_input_supported)
    st.markdown("</div>", unsafe_allow_html=True)


# —————————————————————————————
# Audio Processing & Debate Logic
# —————————————————————————————
def handle_audio_input(audio_data, live_supported: bool):
    """
    Process user audio (live or uploaded), transcribe, generate AI response,
    and update chat_history accordingly.
    """
    if live_supported and audio_data:
        file_path = os.path.join(UPLOAD_DIR, "mic_recording.wav")
        with open(file_path, "wb") as f:
            audio_bytes = audio_data.read()
            f.write(audio_bytes)

        st.session_state.listening = False

        # VAD for future auto-cutoff
        try:
            process_audio_for_vad(audio_bytes)
        except Exception as e:
            st.warning(f"VAD error: {e}")

        # Transcription
        with st.spinner("Transcribing..."):
            user_text = transcribe_audio(file_path, language="en")

        _process_user_text(user_text, audio_bytes)

    elif not live_supported and audio_data:
        # Uploaded file path & bytes
        path = os.path.join(UPLOAD_DIR, audio_data.name)
        with open(path, "wb") as f:
            f.write(audio_data.getbuffer())
        audio_bytes = audio_data.getvalue()

        st.session_state.listening = False
        with st.spinner("Transcribing..."):
            user_text = transcribe_audio(path, language="en")

        _process_user_text(user_text, audio_bytes)


def _process_user_text(user_text: str, audio_bytes: bytes):
    """
    Shared logic for handling new user text:
    - Deduplication
    - Adding to history
    - Generating & appending AI response
    """
    if not user_text or user_text == st.session_state.last_user_input:
        st.session_state.listening = True
        return

    st.session_state.last_user_input = user_text
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_text,
        "audio": audio_bytes
    })

    # Generate AI response
    with st.spinner("AI is responding..."):
        if len(st.session_state.chat_history) == 1:
            context = f"Topic: {st.session_state.debate_topic}. User's opening argument: {user_text}"
        else:
            context = user_text

        bot_text = llm_response_medical_debate(
            context,
            debate_side=st.session_state.debate_side,
            debate_round=len(st.session_state.chat_history)//2 + 1,
        )
        bot_audio = text_to_speech(text=bot_text, voice_id=st.session_state.voice_id)

    st.session_state.chat_history.append({
        "role": "bot",
        "text": bot_text,
        "audio": bot_audio
    })

    # Auto-listen for next turn
    if st.session_state.auto_listen:
        st.session_state.listening = True

    st.rerun()


# —————————————————————————————
# Footer & URL Reset
# —————————————————————————————
def handle_footer_and_reset():
    """Render footer captions and handle URL-based chat reset."""
    if st.session_state.debate_started:
        st.caption("Use Space bar to start/stop recording.")
        st.caption("Click 'New Debate' to restart.")

    if st.query_params.get("clear_chat"):
        for key in ["debate_started", "debate_topic", "chat_history", "listening", "last_user_input", "response", "transcript"]:
            st.session_state[key] = False if isinstance(st.session_state.get(key), bool) else None
        st.rerun()


# —————————————————————————————
# Main
# —————————————————————————————
def main():
    """Main entry point for the Streamlit app."""
    ensure_upload_dir()
    init_session_state()
    setup_page()
    voice_map = load_voices()

    if not st.session_state.debate_started:
        render_setup_panel(voice_map)
    else:
        render_debate_interface()

    handle_footer_and_reset()


if __name__ == "__main__":
    main()
