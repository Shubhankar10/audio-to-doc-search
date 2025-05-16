# app/utils.py
import base64
import streamlit as st

def get_custom_css():
    """
    Return custom CSS for styling the debate app
    """
    return """
    <style>
        /* Custom animation for the listening indicator */
        @keyframes pulse {
            0% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.1); opacity: 1; }
            100% { transform: scale(1); opacity: 0.8; }
        }
        
        /* Voice wave animation for the bars */
        @keyframes voice-wave {
            0% { transform: scaleY(0.5); fill: rgba(255, 75, 75, 0.6); }
            50% { transform: scaleY(0.8); fill: rgba(255, 75, 75, 0.9); }
            100% { transform: scaleY(1.0); fill: rgba(255, 75, 75, 1); }
        }
        
        /* CSS for better Siri-like appearance */
        .voice-bar {
            fill: #ff4b4b;
            transform-origin: bottom;
        }
        
        .listening-indicator {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 10px 0;
        }
        
        .pulse-dot {
            height: 20px;
            width: 20px;
            border-radius: 50%;
            background-color: #ff4b4b;
            margin: 0 5px;
            animation: pulse 1.5s infinite;
        }
        
        .pulse-dot:nth-child(2) {
            animation-delay: 0.3s;
        }
        
        .pulse-dot:nth-child(3) {
            animation-delay: 0.6s;
        }
        
        /* Custom message bubbles */
        .message-bubble {
            padding: 15px;
            border-radius: 20px;
            margin-bottom: 15px;
            max-width: 85%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .message-bubble:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .user-bubble {
            background-color: #e1f5fe;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }
        
        .ai-bubble {
            background-color: #f5f5f5;
            margin-right: auto;
            border-bottom-left-radius: 5px;
        }
        
        /* Make the app feel more modern */
        .main-container {
            padding: 1.2rem;
            border-radius: 12px;
            background-color: white;
            box-shadow: 0 6px 12px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        
        /* Button customizations - more native-app like */
        .custom-button {
            border-radius: 25px;
            padding: 12px 25px;
            font-weight: 500;
            border: none;
            cursor: pointer;
            transition: all 0.2s;
            background-image: linear-gradient(135deg, #3a8dde, #5b6bda);
            color: white;
        }
        
        .custom-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        /* Override Streamlit's default button styles */
        .stButton button {
            border-radius: 25px !important;
            font-weight: 500 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        }
        
        /* Keyboard shortcut indicator */
        .keyboard-shortcut {
            background-color: #f5f5f5;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.8em;
            margin-left: 5px;
            border: 1px solid #e0e0e0;
        }
        
        /* Voice controls */
        .voice-controls {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 15px 0;
        }
        
        /* Enhance the chat container */
        #chat-container {
            border-radius: 8px;
            border: 1px solid #f0f0f0;
            background-color: #fafafa;
            padding: 15px;
            scroll-behavior: smooth;
        }
        
        /* Style for header areas */
        .header-area {
            background-color: #f9f9f9;
            border-radius: 8px 8px 0 0;
            padding: 10px 15px;
            border-bottom: 1px solid #eaeaea;
        }
        
        /* App status indicator */
        .app-status {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-listening {
            background-color: #4CAF50;
            box-shadow: 0 0 5px #4CAF50;
            animation: pulse 1.5s infinite;
        }
        
        .status-thinking {
            background-color: #FFC107;
            box-shadow: 0 0 5px #FFC107;
            animation: pulse 1.5s infinite;
        }
        
        /* Better form controls */
        .stTextInput input, .stSelectbox select {
            border-radius: 8px !important;
            border: 1px solid #e0e0e0 !important;
            padding: 10px !important;
        }
    </style>
    """

def autoplay_audio(audio_bytes):
    """
    Auto-play the audio without requiring user interaction
    """
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def render_listening_animation():
    """
    Display an enhanced animated listening indicator with voice wave visualization
    """
    html = """
    <div class="listening-indicator">
        <div class="pulse-dot"></div>
        <div class="pulse-dot"></div>
        <div class="pulse-dot"></div>
    </div>
    <div style="text-align: center; margin-bottom: 15px;">
        Listening... (Press Space to stop)
    </div>
    <div style="text-align: center; margin-top: 5px;">
        <svg width="120" height="30" viewBox="0 0 120 30">
            <g transform="translate(60, 15)">
                <rect class="voice-bar" x="-50" y="-8" width="5" height="16" rx="2" style="animation: voice-wave 0.5s infinite alternate; animation-delay: 0.0s;"></rect>
                <rect class="voice-bar" x="-40" y="-10" width="5" height="20" rx="2" style="animation: voice-wave 0.5s infinite alternate; animation-delay: 0.1s;"></rect>
                <rect class="voice-bar" x="-30" y="-6" width="5" height="12" rx="2" style="animation: voice-wave 0.5s infinite alternate; animation-delay: 0.2s;"></rect>
                <rect class="voice-bar" x="-20" y="-12" width="5" height="24" rx="2" style="animation: voice-wave 0.5s infinite alternate; animation-delay: 0.3s;"></rect>
                <rect class="voice-bar" x="-10" y="-8" width="5" height="16" rx="2" style="animation: voice-wave 0.5s infinite alternate; animation-delay: 0.4s;"></rect>
                <rect class="voice-bar" x="0" y="-14" width="5" height="28" rx="2" style="animation: voice-wave 0.5s infinite alternate; animation-delay: 0.5s;"></rect>
                <rect class="voice-bar" x="10" y="-9" width="5" height="18" rx="2" style="animation: voice-wave 0.5s infinite alternate; animation-delay: 0.4s;"></rect>
                <rect class="voice-bar" x="20" y="-10" width="5" height="20" rx="2" style="animation: voice-wave 0.5s infinite alternate; animation-delay: 0.3s;"></rect>
                <rect class="voice-bar" x="30" y="-7" width="5" height="14" rx="2" style="animation: voice-wave 0.5s infinite alternate; animation-delay: 0.2s;"></rect>
                <rect class="voice-bar" x="40" y="-9" width="5" height="18" rx="2" style="animation: voice-wave 0.5s infinite alternate; animation-delay: 0.1s;"></rect>
            </g>
        </svg>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_message_bubbles(chat_history):
    """
    Render chat messages with enhanced message bubbles
    """
    for msg in chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="message-bubble user-bubble">
                    <strong>You</strong>
                    <p>{msg["text"]}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="message-bubble ai-bubble">
                    <strong>AI</strong>
                    <p>{msg["text"]}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
