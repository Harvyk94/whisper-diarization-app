# Whisper Transcription & Diarization App

This Streamlit application allows you to transcribe audio files and live audio streams using OpenAI's Whisper model and perform speaker diarization using WhisperX.

## Features

*   **File Upload & Live Recording:** Transcribe audio from uploaded files (MP3, WAV, M4A, etc.) or directly from your microphone.
*   **Whisper Transcription:** Utilizes various Whisper models (tiny, base, small, medium, large-v2, large-v3).
*   **Speaker Diarization:** Identifies different speakers in the audio using `pyannote.audio`.
*   **Timestamped & Labeled Output:** Displays transcription with speaker labels (e.g., `SPEAKER_00 [HH:MM:SS]:`) and configurable periodic timestamps.
*   **Filler Word Highlighting:** Optionally highlights common filler words (um, uh, like, you know, etc.) in different colors in the displayed transcript and Word download. Includes a legend.
*   **Multiple Download Formats:** Download the transcription results as:
    *   HTML (with speaker colors and filler highlighting)
    *   Microsoft Word (.docx) (with speaker colors and filler highlighting)
    *   Plain Text (.txt) (clean transcript with speaker/timestamp labels)
    *   CSV (.csv) (columns: `Timestamp [Start - End]`, `Speaker`, `Transcript`)
*   **GPU Acceleration:** Supports CUDA or MPS (Apple Silicon) if available, with automatic CPU fallback.
*   **Live Audio Level Indicator:** Shows microphone input volume during recording.

## Prerequisites

*   **Python 3.8+**
*   **`ffmpeg`:** This is required by Whisper and WhisperX.
    *   **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    *   **Windows (using Chocolatey):** `choco install ffmpeg`
    *   **Windows (Manual/No Admin):** Download the binaries from the official [FFmpeg website](https://ffmpeg.org/download.html) (e.g., from gyan.dev), extract them to a folder, and either run `ffmpeg.exe` from the `bin` subfolder or add that `bin` subfolder to your User PATH environment variable.
*   **Hugging Face Account & Token:**
    *   Create an account on [Hugging Face](https://huggingface.co/).
    *   Generate an access token ([Settings -> Access Tokens](https://huggingface.co/settings/tokens)).
    *   You **must** accept the user conditions for the following models on the Hugging Face website *while logged in*:
        *   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
        *   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (dependency of diarization)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Harvyk94/whisper-diarization-app.git
    cd whisper-diarization-app
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you don't have a `requirements.txt` file yet, you can create one by running `pip freeze > requirements.txt` after installing the necessary packages like `streamlit`, `whisperx`, `torch`, `python-docx`, `pandas`, `streamlit-webrtc`, `numpy`, `soundfile`, etc.)*
    *(M1/M2/M3 Mac users: Ensure PyTorch is installed correctly for MPS acceleration. The `requirements.txt` should handle this, but refer to the [PyTorch website](https://pytorch.org/) if issues arise.)*

4.  **Set your Hugging Face token:**
    *   Open the `app.py` file.
    *   Find the line `HF_TOKEN = "YOUR_HF_TOKEN"` (or similar placeholder).
    *   **Replace `"YOUR_HF_TOKEN"`** with your actual Hugging Face access token.
    ```python
    # Example in app.py
    HF_TOKEN = "hf_YourActualTokenGoesHere"
    ```

## Usage

1.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
2.  **Open your web browser** to the local URL provided by Streamlit (usually `http://localhost:8501`).
3.  **Configure Settings (Sidebar):**
    *   Select the desired **Whisper Model**. Larger models are more accurate but slower and require more resources.
    *   Choose the **Timestamp Frequency**.
    *   Optionally adjust the minimum/maximum expected speakers for diarization.
4.  **Choose a Tab:**
    *   **⬆️ Upload File:**
        *   Click "Browse files" to upload an audio file.
        *   Click "🎯 Transcribe & Diarize File".
        *   Wait for processing. Results appear in sub-tabs ("Speaker Transcript (Highlighted)" and "Speaker Transcript (Clean)").
        *   Download buttons for HTML, Word, CSV are available in the "Highlighted" tab.
        *   Download button for TXT is available in the "Clean" tab.
    *   **🎙️ Live Recording:**
        *   Click the ▶️ button inside the component box to activate the microphone stream.
        *   Click "🎙️ Start Recording". The mic level indicator will activate.
        *   Speak clearly.
        *   Click "🛑 Stop Recording".
        *   Wait for processing. The transcript will appear below.
        *   Download buttons for TXT and CSV formats will be available below the transcript. 