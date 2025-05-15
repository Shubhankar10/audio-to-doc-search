# app/tts_elevenlabs.py

import requests
from app.instance.config import ELEVENLABS_API_KEY

def list_voices() -> dict:
    """
    Fetch all available ElevenLabs voices.
    Returns a dict containing 'voices' list with 'voice_id' and 'name'.
    """
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def text_to_speech(
    text: str,
    voice_id: str,
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128"
) -> bytes:
    """
    Convert `text` into speech using ElevenLabs TTS Convert endpoint.
    Returns raw audio bytes (MP3) on success, or empty bytes on failure.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    params = {"output_format": output_format}
    payload = {
        "text": text,
        "model_id": model_id
    }

    resp = requests.post(url, headers=headers, params=params, json=payload)
    if resp.status_code == 200:
        return resp.content
    else:
        print(f"[ERROR] ElevenLabs TTS failed ({resp.status_code}): {resp.text}")
        return b""
