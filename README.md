# Whisper Transcription App

This Streamlit application allows you to transcribe audio files using OpenAI's Whisper model and perform speaker diarization using WhisperX.

## Features

*   Upload audio files (MP3, WAV, M4A, etc.).
*   Transcribe audio using various Whisper models (tiny, base, small, medium, large).
*   Perform speaker diarization to identify different speakers.
*   Display transcription with speaker labels and timestamps.
*   Download the transcription as a text file.
*   Utilizes `pyannote.audio` for speaker diarization (requires Hugging Face token).
*   Supports GPU acceleration (CUDA or MPS if available).

## Prerequisites

*   Python 3.8+
*   `ffmpeg`: This is required by Whisper and WhisperX. You can install it using:
    *   **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    *   **Windows (using Chocolatey):** `choco install ffmpeg`
    *   Alternatively, download from the official [FFmpeg website](https://ffmpeg.org/download.html).
*   A Hugging Face account and an access token ([create one here](https://huggingface.co/settings/tokens)). You need to accept the user conditions for the following models:
    *   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
    *   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Installation

1.  **Clone the repository (or download the files):**
    ```bash
    # Replace <your-repo-url> with the actual URL after you create the repo
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you are on an M1/M2/M3 Mac and want MPS acceleration, PyTorch should be installed correctly by the requirements file. If you encounter issues, you might need to install it manually following instructions on the [PyTorch website](https://pytorch.org/).*

4.  **Set your Hugging Face token:**
    Create a file named `.env` in the project root directory and add your token:
    ```
    HUGGINGFACE_TOKEN=your_hf_token_here
    ```
    Alternatively, you can set it as an environment variable:
    ```bash
    export HUGGINGFACE_TOKEN='your_hf_token_here'
    ```

## Usage

1.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your web browser** to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Upload an audio file** using the file uploader.

4.  **Select the desired Whisper model size.** Larger models are more accurate but require more resources and time.

5.  **Click the "Transcribe Audio" button.**

6.  Wait for the transcription and diarization process to complete.

7.  View the results, including the transcription with speaker labels and timestamps.

8.  Optionally, download the transcription as a `.txt` file. 