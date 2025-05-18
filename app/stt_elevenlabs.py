# app/stt_elevenlabs.py

import requests
from app.instance.config import ELEVENLABS_API_KEY

def transcribe_audio(audio_path: str, language: str = "en") -> str:
    """
    Transcribes an audio file using the ElevenLabs Speech-to-Text API.

    Args:
        audio_path (str): The path to the audio file to be transcribed.
        language (str, optional): The language code (ISO-639-1 or ISO-639-3) for transcription. Defaults to "en".

    Returns:
        str: The transcribed text if successful, otherwise an error message.
    """
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}

    data = {
        "model_id": "scribe_v1",        # required
        "language_code": language       # ISO-639-1 or ISO-639-3, e.g. "en" or "eng" :contentReference[oaicite:0]{index=0}
    }
    files = {"file": open(audio_path, "rb")}

    resp = requests.post(url, headers=headers, data=data, files=files)
    if resp.status_code == 200:
        return resp.json().get("text", "")
    else:
        print("[ERROR] ElevenLabs STT:", resp.status_code, resp.text)
        return "Transcription failed."
