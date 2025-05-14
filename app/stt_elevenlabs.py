# app/stt_elevenlabs.py

import requests
from instance import config

def transcribe_audio(audio_path: str) -> str:
    """
    Send the audio file at audio_path to ElevenLabs and return the transcript.
    """
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {
        "xi-api-key": config.ELEVENLABS_API_KEY,    # Your API key header
    }

    # Build multipart form: required model_id + audio file
    data = {
        "model_id": "scribe_v1"                     # required :contentReference[oaicite:3]{index=3}
    }
    files = {
        "file": open(audio_path, "rb")              # must be named "file" :contentReference[oaicite:4]{index=4}
    }

    resp = requests.post(url, headers=headers, data=data, files=files)
    if resp.status_code == 200:
        return resp.json().get("text", "")
    else:
        # Print raw error for debugging
        print("[ERROR] ElevenLabs API returned:", resp.status_code, resp.text)
        return "Transcription failed."
