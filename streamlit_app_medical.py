# Enhanced Medical Debate App with Voice Interface (Siri-like)

import os
import time
import streamlit as st
import numpy as np
import threading
import streamlit.components.v1 as components
from app.stt_elevenlabs import transcribe_audio
from app.tts_elevenlabs import list_voices, text_to_speech
from app.rag_pipeline import llm_response_medical_debate
from app.utils import get_custom_css, autoplay_audio, render_listening_animation, render_message_bubbles
from app.vad import VoiceActivityDetector

# —————————————————————————————
# Helper functions
# —————————————————————————————

def on_silence_detected():
    """Called when silence is detected after speech"""
    if st.session_state.recording:
        # Stop the recording - need to trigger the Record button click
        js_code = """
        <script>
        function stopRecording() {
            const recordButton = document.querySelector('button[aria-label="Record"]');
            if (recordButton) {
                recordButton.click();
            }
        }
        stopRecording();
        </script>
        """
        components.html(js_code, height=0)
        st.session_state.recording = False

def process_audio_for_vad(audio_data, rate=44100):
    """Process audio data to detect silence"""
    vad = VoiceActivityDetector(
        silence_threshold=0.03, 
        silence_duration=st.session_state.vad_timeout,
        on_silence_callback=on_silence_detected
    )
    # Convert to numpy array if needed
    if not isinstance(audio_data, np.ndarray):
        try:
            audio_data = np.frombuffer(audio_data, dtype=np.float32)
        except:
            return False
            
    return vad.process_audio(audio_data)

# —————————————————————————————
# Setup
# —————————————————————————————

st.set_page_config(
    page_title="Medical Voice Debate", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Add keyboard shortcut handling with JavaScript
keyboard_js = """
<script>
// Enhanced keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Space bar for starting/stopping voice recording
    if (e.code === 'Space' && !e.repeat && !e.target.matches('input, textarea')) {
        e.preventDefault();
        
        // Find the mic button and click it
        const micButton = document.querySelector('button[aria-label="Record"]');
        if (micButton) {
            micButton.click();
        }
    }
    
    // Esc key can also stop recording
    if (e.code === 'Escape' && !e.repeat) {
        const micButton = document.querySelector('button[aria-label="Record"].recording');
        if (micButton) {
            micButton.click();
        }
    }
    
    // F1 key to show/hide help
    if (e.code === 'F1') {
        e.preventDefault();
        // Toggle the help expander
        const helpExpander = document.querySelector('details[data-testid="stExpander"]');
        if (helpExpander) {
            helpExpander.open = !helpExpander.open;
        }
    }
});

// Add a class to the record button when it's active
const observeRecordButton = () => {
    // Add mutation observer to detect when recording starts/stops
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'attributes' && mutation.attributeName === 'aria-label') {
                const button = mutation.target;
                if (button.getAttribute('aria-label') === 'Stop') {
                    button.classList.add('recording');
                    // Signal to Streamlit that we're recording
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: true
                    }, '*');
                } else {
                    button.classList.remove('recording');
                    // Signal to Streamlit that we're done recording
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: false
                    }, '*');
                }
            }
        });
    });
    
    // Start observing record button
    setTimeout(() => {
        const recordButton = document.querySelector('button[aria-label="Record"]');
        if (recordButton) {
            observer.observe(recordButton, { attributes: true });
        } else {
            // Try again later if button not found
            setTimeout(observeRecordButton, 1000);
        }
    }, 1000);
};

observeRecordButton();

// Add auto-scroll to chat container
function scrollChatToBottom() {
    const chatContainer = document.getElementById('chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    setTimeout(scrollChatToBottom, 800); // Keep checking
}
scrollChatToBottom();
</script>
"""
components.html(keyboard_js, height=0)

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state with more options
if "response" not in st.session_state:
    st.session_state.response = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "debate_topic" not in st.session_state:
    st.session_state.debate_topic = ""
if "debate_side" not in st.session_state:
    st.session_state.debate_side = "against"  # AI will argue against by default
if "debate_started" not in st.session_state:
    st.session_state.debate_started = False
if "listening" not in st.session_state:
    st.session_state.listening = False
if "recording" not in st.session_state:
    st.session_state.recording = False
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = None
if "voice_id" not in st.session_state:
    st.session_state.voice_id = None
if "voice_name" not in st.session_state:
    st.session_state.voice_name = None
if "auto_listen" not in st.session_state:
    st.session_state.auto_listen = True
if "vad_timeout" not in st.session_state:
    st.session_state.vad_timeout = 2.0  # seconds of silence to auto-stop recording
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = None
if "should_stop_recording" not in st.session_state:
    st.session_state.should_stop_recording = False

# Fetch TTS voices once
try:
    voices = list_voices().get("voices", [])
    voice_map = {v["name"]: v["voice_id"] for v in voices}
    if not st.session_state.voice_id:  # Only set default on first load
        default_voice_name = list(voice_map.keys())[0]
        st.session_state.voice_id = voice_map[default_voice_name]
        st.session_state.voice_name = default_voice_name
except Exception as e:
    st.error(f"Error loading voices: {str(e)}")
    voices = []
    voice_map = {"Default": "default"}
    st.session_state.voice_id = "default"
    st.session_state.voice_name = "Default"

# —————————————————————————————
# Main UI
# —————————————————————————————

col1, col2 = st.columns([1, 4])
with col1:
    # Logo
    st.image("sit-data/logo.png", width=80)

with col2:
    st.title("🎤 Medical Voice Assistant")
    st.markdown("<p style='margin-top:-15px;'>An AI debate partner powered by voice - just like talking to Siri</p>", unsafe_allow_html=True)

# Debate setup (only show if debate not started)
if not st.session_state.debate_started:
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("### Start a New Debate")
    
    # Topic selection
    debate_topic = st.text_input(
        "Enter a medical debate topic:",
        placeholder="e.g., 'Artificial intelligence will replace doctors in the next decade'"
    )
    
    # AI position selection
    col1, col2 = st.columns(2)
    
    with col1:
        ai_side = st.radio(
            "AI should argue:",
            options=["FOR", "AGAINST"],
            horizontal=True
        )
    
    with col2:
        # Voice selection
        selected_voice = st.selectbox(
            "AI voice:",
            options=list(voice_map.keys()),
            index=list(voice_map.keys()).index(st.session_state.voice_name) if st.session_state.voice_name in voice_map else 0,
            key="voice_select"
        )
        # Update voice ID when selection changes
        if selected_voice != st.session_state.voice_name:
            st.session_state.voice_id = voice_map[selected_voice]
            st.session_state.voice_name = selected_voice
    
    # Settings
    with st.expander("Advanced Settings"):
        st.checkbox("Auto-start listening after AI responds", value=st.session_state.auto_listen, key="auto_listen_setting")
        vad_timeout = st.slider("Seconds of silence before auto-stopping recording", 1.0, 5.0, st.session_state.vad_timeout, 0.5)
        if vad_timeout != st.session_state.vad_timeout:
            st.session_state.vad_timeout = vad_timeout
    
    # Start debate button with better styling
    if st.button("Start Debate", type="primary", use_container_width=True):
        if debate_topic:
            st.session_state.debate_topic = debate_topic
            st.session_state.debate_side = ai_side.lower()
            st.session_state.debate_started = True
            st.session_state.debate_round = 1
            st.session_state.chat_history = []
            st.session_state.auto_listen = st.session_state.get("auto_listen_setting", True)
            st.rerun()
        else:
            st.error("Please enter a debate topic to begin.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add some tips for first-time users
    with st.expander("Tips for using the Voice Debate Assistant"):
        st.markdown("""
        - Press the **Space Bar** to start/stop recording (keyboard shortcut)
        - The app will automatically start listening after each AI response
        - You can customize the AI's voice in the settings
        - Try to speak clearly and concisely for best results
        - You'll see visual indicators when the app is listening
        """)

# Debate interface (show only if debate has started)
if st.session_state.debate_started:
    # Chat container with fixed height and better styling
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    
    # Topic header
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div>
                <h3 style="margin: 0;">{st.session_state.debate_topic}</h3>
                <p style="margin: 0; color: #666;">AI is arguing <strong>{st.session_state.debate_side.upper()}</strong></p>
            </div>
            <div>
                <span class="keyboard-shortcut">Press Space to speak</span>
            </div>
        </div>
        <hr>
        """, 
        unsafe_allow_html=True
    )
    
    # Display chat in a scrollable container
    chat_container = st.container()
    
    # Fixed height chat container
    chat_container.markdown(
        """
        <div style="height: 350px; overflow-y: auto; margin-bottom: 20px; padding-right: 10px;" id="chat-container">
        """,
        unsafe_allow_html=True
    )
    
    # Render chat history with enhanced bubbles
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            chat_container.markdown(
                f"""
                <div class="message-bubble user-bubble">
                    <strong>You</strong>
                    <p>{msg["text"]}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            # Play user audio
            if msg == st.session_state.chat_history[-1] and msg["audio"]:
                chat_container.audio(msg["audio"], format="audio/wav")
        else:
            chat_container.markdown(
                f"""
                <div class="message-bubble ai-bubble">
                    <strong>AI</strong>
                    <p>{msg["text"]}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            # Auto-play the most recent AI message
            if msg == st.session_state.chat_history[-1] and msg["audio"]:
                # Use auto-play for the audio
                with chat_container:
                    st.audio(msg["audio"], format="audio/mp3", autoplay=True)
                    # Set the state to listening mode after AI completes its response
                    if st.session_state.auto_listen:
                        st.session_state.listening = True
    
    chat_container.markdown("</div>", unsafe_allow_html=True)
    
    # Voice input section with better UI
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Dynamic listening state visualization
    if st.session_state.listening:
        # Show animated listening indicator
        render_listening_animation()
    else:
        st.markdown(
            """
            <div style="text-align: center; margin: 15px 0;">
                <div style="color: #666;">AI is thinking...</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Voice input controls
    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        if st.button("⏹️ New Debate", type="secondary"):
            st.session_state.debate_started = False
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        # Use audio_input if available (preferred) or file uploader as fallback
        try:
            # This component will also set st.session_state.recording via JavaScript
            recording_status = st.empty()
            audio_recording = st.audio_input(
                "Speak your argument...",
                key="voice_input",
                label_visibility="collapsed",
                on_change=lambda: setattr(st.session_state, 'recording', not st.session_state.recording)
            )
            
            # Add a more descriptive prompt under the button
            st.markdown(
                "<div style='text-align: center; font-size: 0.8em; color: #666;'>"
                "Press Space to speak, releases automatically when you pause"
                "</div>",
                unsafe_allow_html=True
            )
            
            audio_input_supported = True
        except AttributeError:
            st.warning("Live voice input not supported in your Streamlit version. Please upload a recording instead.")
            audio_recording = None
            audio_file = st.file_uploader(
                "Upload your voice recording (MP3/WAV):",
                type=["mp3", "wav"],
                key="upload_file"
            )
            audio_input_supported = False
    
    with col3:
        if len(st.session_state.chat_history) > 0:
            if st.button("🔄 Replay Last", type="secondary"):
                # Find the last AI message and replay it
                for msg in reversed(st.session_state.chat_history):
                    if msg["role"] == "bot" and msg["audio"]:
                        st.audio(msg["audio"], format="audio/mp3", autoplay=True)
                        break
    
    st.markdown("</div>", unsafe_allow_html=True)
      # Process voice input
    if audio_input_supported and audio_recording:
        path = os.path.join(UPLOAD_DIR, "mic_recording.wav")
        
        try:
            # Write the audio to a file
            with open(path, "wb") as f:
                audio_bytes = audio_recording.read()
                f.write(audio_bytes)
            
            # Process audio for VAD (voice activity detection)
            try:
                audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                if len(audio_data) > 0:
                    # Process for silence detection for future auto-cutoff
                    process_audio_for_vad(audio_data)
            except Exception as e:
                st.warning(f"VAD processing error: {str(e)}")
            
            # After getting user input, set listening to false
            st.session_state.listening = False
            
            # Transcribe
            with st.spinner("Transcribing..."):
                user_text = transcribe_audio(path, language="en")
            
            # Make sure it's not a duplicate of the last input and not empty
            if user_text and user_text.strip() and user_text != st.session_state.last_user_input:
                st.session_state.last_user_input = user_text
                
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "text": user_text,
                    "audio": audio_bytes
                })
                
                # Get AI response
                with st.spinner("AI is responding..."):
                    # Add topic context for first message
                    if len(st.session_state.chat_history) == 1:
                        context = f"Topic: {st.session_state.debate_topic}. User's opening argument: {user_text}"
                    else:
                        context = user_text
                    
                    # Generate AI response
                    bot_text = llm_response_medical_debate(
                        context,
                        debate_side=st.session_state.debate_side,
                        debate_round=len(st.session_state.chat_history) // 2 + 1
                    )
                    bot_audio = text_to_speech(text=bot_text, voice_id=st.session_state.voice_id)
                
                # Add AI response to chat history
                st.session_state.chat_history.append({
                    "role": "bot",
                    "text": bot_text,
                    "audio": bot_audio
                })
                
                # After AI has responded, set listening back to true for the next turn if auto listen is enabled
                if st.session_state.auto_listen:
                    st.session_state.listening = True
                
                # Force UI refresh
                st.rerun()
        except Exception as e:
            st.error(f"Error processing recording: {str(e)}")
            # Reset listening state
            st.session_state.listening = True
            
    # Also handle file uploads for systems without audio input
    elif not audio_input_supported and audio_file:
        path = os.path.join(UPLOAD_DIR, audio_file.name)
        with open(path, "wb") as f:
            f.write(audio_file.getbuffer())
        user_audio_bytes = audio_file.getvalue()
        
        # After getting user input, set listening to false
        st.session_state.listening = False
        
        # Transcribe
        with st.spinner("Transcribing..."):
            user_text = transcribe_audio(path, language="en")
        
        # Make sure it's not a duplicate of the last input
        if user_text != st.session_state.last_user_input:
            st.session_state.last_user_input = user_text
            
            # Process as before...
            st.session_state.chat_history.append({
                "role": "user",
                "text": user_text,
                "audio": user_audio_bytes
            })
            
            with st.spinner("AI is responding..."):
                if len(st.session_state.chat_history) == 1:
                    context = f"Topic: {st.session_state.debate_topic}. User's opening argument: {user_text}"
                else:
                    context = user_text
                
                bot_text = llm_response_medical_debate(
                    context,
                    debate_side=st.session_state.debate_side,
                    debate_round=len(st.session_state.chat_history) // 2 + 1
                )
                bot_audio = text_to_speech(text=bot_text, voice_id=st.session_state.voice_id)
            
            st.session_state.chat_history.append({
                "role": "bot",
                "text": bot_text,
                "audio": bot_audio
            })
            
            if st.session_state.auto_listen:
                st.session_state.listening = True
            st.rerun()
        
    # If starting a new debate, show welcome message
    if st.session_state.debate_started and len(st.session_state.chat_history) == 0:
        st.info("Start the debate by speaking your opening argument about the topic.")
        # Set listening mode to true for the first turn
        st.session_state.listening = True

# —————————————————————————————
# Footer information
# —————————————————————————————

if st.session_state.debate_started:
    st.caption("Use the Space bar as a keyboard shortcut to start/stop recording.")
    st.caption("To end this debate and start a new one, click the 'New Debate' button.")

# Clear logic for URL param-based reset
if st.query_params.get("clear_chat"):
    st.session_state.debate_started = False
    st.session_state.debate_topic = ""
    st.session_state.chat_history = []
    st.session_state.listening = False
    st.session_state.last_user_input = None
    st.session_state.response = None
    st.session_state.transcript = None
    st.rerun()
