# audio-to-doc-search
A Streamlit-based application that lets users upload or record audio questions, transcribes them using [ElevenLabs](https://elevenlabs.io), retrieves relevant context from documents using RAG, and generates answers with an LLM via Ollama.


---

## Directory Structure:

### `app/`  
Contains the core application modules:
- **`rag_pipeline.py`** – Loads the vector database, retrieves context chunks, and defines `get_llm_response(query)` to call the LLM.  
- **`llm_ollama.py`** – Handles communication with the Ollama server for LLM inference.  
- **`stt_elevenlabs.py`** – Wraps the ElevenLabs Speech-to-Text API to transcribe uploaded or recorded audio.  
- **`tts_elevenlabs.py`** – Wraps the ElevenLabs Text-to-Speech API to synthesize audio from text responses.  
- **`utils.py`** – Utility functions for configuration loading, file handling, and shared helpers.  
- **`__init__.py`** – Marks `app/` as a Python package.

### `sit-data/`  
Holds sample SIT (System Integration Testing) documents used to build and test the RAG retrieval workflows.

### `vector_context/`  
Stores precomputed vector embeddings and context files for fast similarity search during the RAG process.

### `.gitignore`  
Specifies files and directories (e.g., virtual environments, temporary uploads) that Git should ignore.

### `LICENSE`  
The full text of the MIT License, under which this project is released.

### `README.md`  
This file—provides an overview and concise descriptions of the repository contents.

### `requirements.txt`  
Lists all Python dependencies required to run the application (Streamlit, Requests, ElevenLabs & Ollama SDKs, etc.).

### `streamlit_app.py`  
The main Streamlit application:
1. Audio upload/recording  
2. STT transcription  
3. RAG query to the LLM  
4. TTS playback of the response

### `streamlit_app_medical.py`  
A domain-specific variant of `streamlit_app.py` tailored for medical SIT data and prompts.

### `streamlit_app_sit.py`  
A specialized Streamlit script demonstrating end-to-end queries against the SIT dataset.

---
