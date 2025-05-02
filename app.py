import streamlit as st
import whisperx
import tempfile
import os
import torch
import re
from docx import Document
from docx.shared import RGBColor, Pt
import io
from datetime import timedelta
import pandas as pd
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import queue
import soundfile as sf

# Placeholder for Hugging Face token - replace with your actual token or ensure login
HF_TOKEN = "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" # Replace with your token

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=round(seconds)))

def get_device():
    """Detects the appropriate device (MPS, CUDA, or CPU)."""
    # Check for MPS (Apple Silicon GPU)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
         try:
             # Test MPS device availability
             _ = torch.tensor([1], device="mps")
             return "mps"
         except Exception as e:
             st.warning(f"⚠️ MPS device detected but test failed: {e}. Falling back...")

    # Check for CUDA
    if torch.cuda.is_available():
        return "cuda"

    # Default to CPU
    return "cpu"

@st.cache_resource
def load_whisperx_model(model_name="base"):
    """Loads the WhisperX model."""
    try:
        device = get_device()
        # Use float16 for CUDA/MPS if available, else int8 for CPU
        compute_type = "float16" if device in ["cuda", "mps"] else "int8"
        model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
        st.success(f"WhisperX model '{model_name}' loaded successfully on {device.upper()} (Compute: {compute_type}).")
        return model
    except Exception as e:
        st.error(f"Error loading WhisperX model: {e}")
        st.warning("⚠️ Falling back to CPU if CUDA/MPS loading failed previously.")
        # Attempt CPU fallback explicitly if needed
        try:
             model = whisperx.load_model(model_name, device="cpu", compute_type="int8")
             st.success(f"WhisperX model '{model_name}' loaded successfully on CPU.")
             return model
        except Exception as fallback_e:
            st.error(f"Fatal error loading WhisperX model on CPU: {fallback_e}")
            return None

@st.cache_resource
def load_diarization_model():
    """Loads the Pyannote diarization pipeline."""
    diarize_model = None # Initialize to None
    try:
        device = get_device() # Get the detected device (might be 'cpu' after fallback)
        st.info(f"Attempting to load diarization model on: {device.upper()}") # Add info

        # Check if token is placeholder
        if HF_TOKEN == "YOUR_HF_TOKEN":
             st.warning("Using default Pyannote model without authentication. For potentially better results, provide a Hugging Face token.")
             diarize_model = whisperx.DiarizationPipeline(device=device)
        else:
             diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

        # Explicit check after loading
        if diarize_model is None:
             raise ValueError("DiarizationPipeline returned None.")

        st.success(f"Pyannote diarization model loaded successfully on {device.upper()}.")
        return diarize_model

    except Exception as e:
        st.error(f"Error loading diarization model: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}") # Log full traceback
        if "401 Client Error" in str(e):
             st.error("Authentication failed. Please ensure your Hugging Face token is correct and you have accepted the user conditions for pyannote/speaker-diarization-3.1 on Hugging Face.")
        # Add specific check for the 'NoneType' error observed
        if isinstance(e, AttributeError) and "'NoneType' object has no attribute 'to'" in str(e):
            st.warning("The diarization pipeline failed internally, possibly during model download or device placement.")

        return None # Ensure None is returned on any error

def process_audio_with_diarization(audio_file, whisper_model, diarize_model, timestamp_freq):
    """Process audio file, transcribe, align, diarize, and format output.
    Returns:
        tuple: (highlighted_html, clean_text, segments_data)
            segments_data is a list of dicts: [{'start': float, 'end': float, 'speaker': str, 'text': str}]
    """
    if whisper_model is None or diarize_model is None:
        st.error("Models not loaded properly. Cannot process audio.")
        return "Error: Models not loaded.", "Error: Models not loaded.", [] # Return empty list

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name

    try:
        device = get_device()
        batch_size = 16 # Adjust based on MPS memory
        # Compute type already handled in model loading, but specify for alignment
        compute_type = "float16" if device in ["cuda", "mps"] else "int8"

        # 1. Transcribe with WhisperX (includes alignment model loading)
        st.info(f"Loading alignment model...") # Removed device mention
        # Pass device to align model loading - FORCE CPU FOR ALIGNMENT
        alignment_device = 'cpu'
        st.info(f"Loading alignment model explicitly on {alignment_device.upper()}")
        model_a, metadata = whisperx.load_align_model(language_code="en", device=alignment_device)
        st.info(f"Transcribing audio on {device.upper()}...")
        audio = whisperx.load_audio(tmp_path)
        result = whisper_model.transcribe(audio, batch_size=batch_size)

        st.info(f"Aligning transcript on {alignment_device.upper()}...") # Use alignment_device
        # Ensure align also uses the right device - FORCE CPU FOR ALIGNMENT
        result = whisperx.align(result["segments"], model_a, metadata, audio, device=alignment_device, return_char_alignments=False)

        # 2. Assign speaker labels
        st.info(f"Assigning speaker labels on {device.upper()}...") # Use original device
        # Diarization model should already be on the correct device from loading
        diarize_segments = diarize_model(audio)

        # Ensure diarize_segments is usable - might need adjustment based on pyannote version
        # Example check: if isinstance(diarize_segments, Annotation): convert or extract df

        result = whisperx.assign_word_speakers(diarize_segments, result)
        segments = result["segments"] # Segments now have word-level timestamps and speakers

        # 3. Format output
        processed_text_html = []
        processed_text_clean = []
        current_speaker = None
        last_timestamp_marker = -timestamp_freq # Initialize to ensure first timestamp shows
        segments_data_for_csv = [] # Initialize list for structured data

        speaker_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ] # Colors for different speakers

        for i, segment in enumerate(segments):
            if not segment.get('words'):
                continue # Skip segments without word details if any

            segment_start_time = segment["start"]

            # Add timestamp marker if needed
            if segment_start_time // timestamp_freq > last_timestamp_marker // timestamp_freq:
                 ts_str = format_timestamp(segment_start_time)
                 processed_text_html.append(f"<br><b>[{ts_str}]</b><br>")
                 processed_text_clean.append(f"\n[{ts_str}]\n")
                 last_timestamp_marker = segment_start_time

            segment_speaker = segment.get("speaker", "UNKNOWN")
            if segment_speaker != current_speaker:
                 # Add timestamp when speaker changes
                 speaker_start_ts = format_timestamp(segment["start"])
                 processed_text_html.append(f'<br><b>{segment_speaker} [{speaker_start_ts}]:</b> ') # Added timestamp
                 processed_text_clean.append(f'\n{segment_speaker} [{speaker_start_ts}]: ') # Added timestamp
                 current_speaker = segment_speaker

            speaker_color_index = int(segment_speaker.split('_')[1]) % len(speaker_colors) if "SPEAKER_" in segment_speaker else 0
            speaker_color = speaker_colors[speaker_color_index]

            # Process words within the segment
            segment_html = []
            segment_clean = []
            segment_words = [] # To collect words for CSV
            for word_info in segment['words']:
                word = word_info['word']
                # Apply filler word highlighting logic here for HTML
                highlighted_word_html, _ = highlight_filler_words(word, inline=True) # Pass inline=True
                segment_html.append(highlighted_word_html)
                segment_clean.append(word)
                segment_words.append(word) # Add word to list

            # Join words for the segment, prepending speaker tag and color
            joined_segment_html = " ".join(segment_html)
            joined_segment_clean = " ".join(segment_clean)
            joined_segment_words = " ".join(segment_words).strip() # Join words for CSV text

            processed_text_html.append(f'<span style="color: {speaker_color}">{joined_segment_html}</span>')
            processed_text_clean.append(joined_segment_clean)

            # Append structured data for this segment
            if joined_segment_words: # Only add if there's text
                segments_data_for_csv.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'speaker': segment_speaker,
                    'text': joined_segment_words
                })


        full_highlighted_text = "".join(processed_text_html).strip().replace("<br>", "\n") # Use \n for cleaner spacing
        full_clean_text = "".join(processed_text_clean).strip()

        # Final cleanup for HTML display
        full_highlighted_text = full_highlighted_text.replace("\n\n","\n").replace("\n ", "\n")


        return full_highlighted_text, full_clean_text, segments_data_for_csv # Return structured data

    except Exception as e:
        st.error(f"Error during transcription/diarization: {str(e)}")
        import traceback
        st.error(traceback.format_exc()) # Print full traceback for debugging
        return "Error during processing.", "Error during processing.", [] # Return empty list on error
    finally:
        os.unlink(tmp_path)


def highlight_filler_words(text, inline=False): # Added inline flag
    """Highlight filler words in the transcript. If inline, returns only highlighted text."""
    filler_patterns = [
        (r'\b(um|uh|ah|er)\b', '#FFD700'),  # Yellow
        (r'\b(like|you know|i mean)\b', '#98FB98'),  # Green
        (r'\b(so|basically|actually|literally)\b', '#87CEFA'),  # Blue
        (r'\b(right|okay|well)\b', '#FFA07A')  # Orange
    ]
    
    highlighted_text = text
    
    for pattern, color in filler_patterns:
        # Ensure pattern handles potential surrounding HTML tags if applied word-by-word later
        # This basic version might have issues if a filler word itself contains HTML
        highlighted_text = re.sub(
            pattern,
            # Use span for HTML highlighting
            lambda m: f'<span style="background-color: {color}">{m.group()}</span>',
            highlighted_text,
            flags=re.IGNORECASE
        )
    
    if inline:
         return highlighted_text, text # Just return modified text for inline use
    else:
         # Original behavior for full text block
         return highlighted_text, text


def create_word_document(text, segments_data=None): # Allow passing structured data
    """Create Word document with timestamps, speakers, and highlighted filler words."""
    doc = Document()
    doc.add_heading('Transcript', 0)

    # Define filler patterns and colors for Word
    filler_patterns_word = [
        (r'\b(um|uh|ah|er)\b', RGBColor(255, 215, 0)), # Yellow
        (r'\b(like|you know|i mean)\b', RGBColor(152, 251, 152)), # Green
        (r'\b(so|basically|actually|literally)\b', RGBColor(135, 206, 250)), # Blue
        (r'\b(right|okay|well)\b', RGBColor(255, 160, 122)) # Orange
    ]
    # Define speaker colors (consistent with HTML)
    speaker_colors_word = [
        RGBColor(31, 119, 180), RGBColor(255, 127, 14), RGBColor(44, 160, 44),
        RGBColor(214, 39, 40), RGBColor(148, 103, 189), RGBColor(140, 86, 75),
        RGBColor(227, 119, 194), RGBColor(127, 127, 127), RGBColor(188, 189, 34),
        RGBColor(23, 190, 207)
    ]

    # Add legend
    legend = doc.add_paragraph()
    legend.add_run('Color Legend:\n').bold = True
    legend.add_run('Yellow: Sound fillers (um, uh, ah, er)\n')
    legend.add_run('Green: Verbal pauses (like, you know, i mean)\n')
    legend.add_run('Blue: Unnecessary modifiers (so, basically, actually, literally)\n')
    legend.add_run('Orange: Transitional words (right, okay, well)\n')
    legend.add_run('Speaker Colors: Assigned sequentially.\n\n') # Added speaker note

    # --- New Processing Logic for Word Doc ---
    # Use the clean text which now includes speaker and timestamp markers
    lines = text.split('\n')
    current_paragraph = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('[') and line.endswith(']'):
            # Timestamp - add as a new paragraph, bold
            current_paragraph = doc.add_paragraph()
            run = current_paragraph.add_run(line)
            run.font.bold = True
        elif ':' in line and any(spk in line for spk in ['SPEAKER_', 'UNKNOWN:']):
            # Speaker line (e.g., "SPEAKER_00: Hello there")
            parts = line.split(':', 1)
            speaker = parts[0].strip()
            content = parts[1].strip() if len(parts) > 1 else ""

            current_paragraph = doc.add_paragraph() # Start new paragraph for speaker text
            speaker_run = current_paragraph.add_run(f"{speaker}: ")
            speaker_run.font.bold = True
            if "SPEAKER_" in speaker:
                 # Extract only the numeric part before the timestamp
                 try:
                     speaker_number_str = speaker.split('_')[1].split(' ')[0]
                     speaker_index = int(speaker_number_str) % len(speaker_colors_word)
                     speaker_run.font.color.rgb = speaker_colors_word[speaker_index]
                 except (IndexError, ValueError):
                     # Fallback if parsing fails (should not happen with expected format)
                     speaker_run.font.color.rgb = RGBColor(128, 128, 128) # Grey
            else: # UNKNOWN speaker
                 speaker_run.font.color.rgb = RGBColor(128, 128, 128) # Grey

            # Process content for filler words
            start_pos = 0
            for pattern, color in filler_patterns_word:
                 # Need to search within the content part only
                 temp_content = ""
                 last_match_end = 0
                 for match in re.finditer(pattern, content, flags=re.IGNORECASE):
                     # Add text before match
                     temp_content += content[last_match_end:match.start()]
                     # Add highlighted match
                     temp_content += f"<{match.group()}|{color}>" # Temporary marker
                     last_match_end = match.end()
                 # Add remaining text
                 temp_content += content[last_match_end:]
                 content = temp_content # Update content with markers for this pattern

            # Now add runs based on markers
            sub_parts = re.split(r'(<[^|>]+?\|[^>]+?>)', content) # Split by markers
            for part in sub_parts:
                 if part.startswith('<') and '|' in part and part.endswith('>'):
                     word, color_str = part[1:-1].split('|', 1)
                     # Find the RGBColor object matching the string representation (or recreate)
                     # This is inefficient, better to pass RGB objects if possible
                     rgb_color = None
                     for _, c in filler_patterns_word:
                         if str(c) in color_str: # Basic check
                             rgb_color = c
                             break
                     run = current_paragraph.add_run(word)
                     if rgb_color:
                         run.font.color.rgb = rgb_color
                 elif part: # Regular text
                     current_paragraph.add_run(part)

        # This simplistic line-by-line processing might misinterpret multi-line speaker segments
        # A more robust approach would use the structured `segments_data` if passed


    return doc

# --- Add CSV Formatting Function ---
def create_csv_data(segments_data):
    """Formats segment data into a CSV string."""
    if not segments_data:
        return ""

    data_for_df = []
    for segment in segments_data:
        start_ts = format_timestamp(segment['start'])
        end_ts = format_timestamp(segment['end'])
        timestamp_range = f"[{start_ts} - {end_ts}]"
        data_for_df.append({
            "Timestamp [Start - End]": timestamp_range,
            "Speaker": segment['speaker'],
            "Transcript": segment['text']
        })

    df = pd.DataFrame(data_for_df)
    return df.to_csv(index=False)

# --- Global variables or Session State for Live Recording ---
# Use session state to hold audio buffer across reruns
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = queue.Queue()
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'live_transcription_results' not in st.session_state:
    st.session_state.live_transcription_results = (None, None, None) # (highlighted, clean, segments_data)
# Add state for audio level
if 'audio_level' not in st.session_state:
    st.session_state.audio_level = 0.0


# Callback function to receive audio frames from streamlit-webrtc
def audio_frame_callback(frame):
    # --- Start Debug Logs ---
    print("DEBUG: audio_frame_callback triggered.") # Print to terminal
    try:
        sound = frame.to_ndarray(format="s16") # s16 for 16-bit signed integer
        print(f"DEBUG: Received frame shape: {sound.shape}, dtype: {sound.dtype}")
    except Exception as e:
        print(f"DEBUG: Error in frame.to_ndarray(): {e}")
        st.session_state.audio_level = 0.0 # Ensure level is 0 on error
        return frame # Exit early if frame conversion fails
    # --- End Debug Logs ---

    if not st.session_state.is_recording:
        # Update level to 0 when not recording
        st.session_state.audio_level = 0.0
        return frame # Do nothing if not recording

    # Calculate RMS for volume level (normalize based on int16 range)
    # Use float32 for calculations to avoid overflow
    sound_float = sound.astype(np.float32)
    rms = np.sqrt(np.mean(sound_float**2))
    # Normalize RMS to roughly 0-1 range (heuristic, adjust max value if needed)
    max_possible_rms = np.sqrt(np.mean(np.array([32767, -32768], dtype=np.float32)**2))
    # Avoid division by zero
    audio_level_normalized = rms / max_possible_rms if max_possible_rms > 0 else 0.0
    # Clip to ensure it stays within 0-1
    st.session_state.audio_level = np.clip(audio_level_normalized, 0.0, 1.0)


    # Whisper works best with mono audio
    if sound.shape[1] > 1:
        sound = np.mean(sound, axis=1, dtype=np.int16)
    st.session_state.audio_buffer.put(sound)
    return frame

# --- Main Application Code --- #

def main():
    st.set_page_config(page_title="Speech-to-Text & Diarization", layout="wide")

    st.title("📝 Speech-to-Text & Diarization App") # Updated title

    # --- Initialize session state for uploaded file results --- #
    if 'uploaded_transcript_hl' not in st.session_state:
        st.session_state.uploaded_transcript_hl = None
    if 'uploaded_transcript_clean' not in st.session_state:
        st.session_state.uploaded_transcript_clean = None
    if 'uploaded_segments_data' not in st.session_state:
        st.session_state.uploaded_segments_data = None
    if 'processed_file_name' not in st.session_state:
         st.session_state.processed_file_name = None

    # --- Sidebar remains the same --- #
    st.sidebar.header("Settings")
    whisper_model_name = st.sidebar.selectbox(
        "Whisper Model",
        options=["tiny", "base", "small", "medium", "large-v2", "large-v3"], # Common Whisper(X) models
        index=1 # Default to base
    )
    timestamp_freq = st.sidebar.selectbox(
        "Timestamp Frequency (approx)",
        options=[30, 60, 120, 300],
        format_func=lambda x: f"Every {x//60} min" if x >= 60 else f"Every {x} sec",
        index=1
    )
    # Note: Diarization might not be very effective on short live snippets
    # Consider disabling or simplifying for live mode if results are poor
    min_speakers = st.sidebar.number_input("Min Speakers (diarization)", min_value=1, value=1)
    max_speakers = st.sidebar.number_input("Max Speakers (diarization, optional)", min_value=1, value=5, step=1)

    # --- Load Models --- #
    # Load models only once
    whisper_model = load_whisperx_model(whisper_model_name)
    diarize_model = load_diarization_model()
    device = get_device()
    st.sidebar.info(f"🖥️ Using device: {device.upper()}")

    # --- Tabs for Upload vs Live --- #
    tab_upload, tab_live = st.tabs(["⬆️ Upload File", "🎙️ Live Recording"])

    # --- Upload Tab --- #
    with tab_upload:
        st.header("Transcribe from Audio File")
        st.markdown("""
        ### Color Legend:
        - Speaker turns are indicated by color and label (e.g., SPEAKER_00).
        - <span style="background-color: #FFD700">Yellow</span>: Sound fillers (um, uh, ah, er)
        - <span style="background-color: #98FB98">Green</span>: Verbal pauses (like, you know, I mean)
        - <span style="background-color: #87CEFA">Blue</span>: Unnecessary modifiers (so, basically, actually, literally)
        - <span style="background-color: #FFA07A">Orange</span>: Transitional words (right, okay, well)
        """, unsafe_allow_html=True)

        audio_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a', 'ogg'])

        if audio_file is not None:
            st.audio(audio_file)

            # Clear previous results if a new file is uploaded
            if st.session_state.processed_file_name != audio_file.name:
                 st.write("DEBUG: New file detected, clearing previous state.") # DEBUG
                 st.session_state.uploaded_transcript_hl = None
                 st.session_state.uploaded_transcript_clean = None
                 st.session_state.uploaded_segments_data = None
                 st.session_state.processed_file_name = None # Reset processed file name

            if st.button("🎯 Transcribe & Diarize File", use_container_width=True):
                if whisper_model and diarize_model:
                    with st.spinner("Processing Uploaded Audio..."):
                        highlighted_transcript, clean_transcript, segments_data = process_audio_with_diarization(
                            audio_file, whisper_model, diarize_model, timestamp_freq
                        )
                        # Store results in session state
                        st.session_state.uploaded_transcript_hl = highlighted_transcript
                        st.session_state.uploaded_transcript_clean = clean_transcript
                        st.session_state.uploaded_segments_data = segments_data
                        st.session_state.processed_file_name = audio_file.name # Mark this file as processed
                        st.write("DEBUG: Processing complete, results stored in session state.") # DEBUG
                        # Rerun to display results immediately after processing
                        st.rerun()
                else:
                    st.error("Models failed to load. Cannot transcribe file.")
                    # Clear potentially stale results on model load failure
                    st.session_state.uploaded_transcript_hl = None
                    st.session_state.uploaded_transcript_clean = None
                    st.session_state.uploaded_segments_data = None
                    st.session_state.processed_file_name = None

            # --- Display results (outside the button 'if', reads from session state) ---
            # Only display if the currently loaded file matches the processed file in state
            if st.session_state.processed_file_name == audio_file.name and st.session_state.uploaded_transcript_hl:
                 st.write("DEBUG: Displaying results from session state.") # DEBUG
                 # Create doc from session state data before showing tabs
                 # Ensure clean text exists before creating doc
                 if st.session_state.uploaded_transcript_clean:
                     doc = create_word_document(st.session_state.uploaded_transcript_clean)
                     doc_buffer = io.BytesIO()
                     doc.save(doc_buffer)
                     doc_buffer.seek(0)
                 else: 
                     doc_buffer = None # Handle case where clean text might be missing

                 # Create CSV data from session state data
                 # Ensure segment data exists
                 if st.session_state.uploaded_segments_data:
                     csv_data = create_csv_data(st.session_state.uploaded_segments_data)
                 else:
                     csv_data = None # Handle case where segment data might be missing
                 
                 # Extract base filename for download buttons
                 base_filename = os.path.splitext(st.session_state.processed_file_name)[0]

                 sub_tab1, sub_tab2 = st.tabs(["Speaker Transcript (Highlighted)", "Speaker Transcript (Clean)"])                 
                 with sub_tab1:
                      st.markdown(st.session_state.uploaded_transcript_hl, unsafe_allow_html=True)
                      col1, col2, col3 = st.columns(3) # Use 3 columns for downloads
                      with col1:
                           # Fix: Replace newline outside f-string
                           highlighted_transcript_html = st.session_state.uploaded_transcript_hl.replace('\n', '<br>')
                           html_data = f"""
                           <!DOCTYPE html>
                           <html> <head> <title>Transcript</title> <style> body {{ font-family: sans-serif; }} span[style*="background-color"] {{ padding: 1px 3px; border-radius: 3px; }} b {{ font-weight: bold; }} </style> </head> <body> {highlighted_transcript_html} </body> </html>
                           """
                           st.download_button(
                               label="📥 Download as HTML",
                               data=html_data,
                               file_name=f"{base_filename}_transcript.html", # Use processed filename
                               mime="text/html",
                               key="download_html_upload" # Add unique key
                           )
                      with col2:
                           if doc_buffer:
                               st.download_button(
                                   label="📥 Download as Word",
                                   data=doc_buffer,
                                   file_name=f"{base_filename}_transcript.docx", # Use processed filename
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                   key="download_word_upload" # Add unique key
                               )
                           else: 
                               st.info("Word doc could not be generated.")
                      with col3: # Add column for CSV download
                           if csv_data is not None:
                               st.download_button(
                                   label="📥 Download as CSV",
                                   data=csv_data,
                                   file_name=f"{base_filename}_transcript.csv", # Use processed filename
                                   mime="text/csv",
                                   key="download_csv_upload" # Add unique key
                               )
                           else:
                               st.info("CSV data could not be generated.")
                 with sub_tab2:
                      st.text_area("Clean Transcript with Speakers", st.session_state.uploaded_transcript_clean, height=300)
                      st.download_button(
                          label="📥 Download Clean Transcript (TXT)",
                          data=st.session_state.uploaded_transcript_clean,
                          file_name=f"{base_filename}_transcript.txt", # Use processed filename
                          mime="text/plain",
                          key="download_txt_upload" # Add unique key
                      )
            # Optional: Add a message if processing was triggered but failed silently
            # elif st.session_state.processed_file_name == audio_file.name:
            #      st.info("Processing completed, but no results to display.")

    # --- Live Recording Tab --- #
    with tab_live:
        st.header("Transcribe from Microphone")
        st.write("Click 'Start Recording', speak, then 'Stop Recording' to transcribe.")
        st.info("Note: Diarization accuracy may be lower on short live recordings.")

        # WebRTC Streamer configuration
        webrtc_ctx = webrtc_streamer(
            key="live-audio",
            mode=WebRtcMode.SENDRECV, # Send and receive audio (needed for processing)
            rtc_configuration={ # Pass directly
                # Use Twilio's STUN server instead
                "iceServers": [{"urls": ["stun:stun.twilio.com:3478"]}]
            },
            media_stream_constraints={ # Pass directly
                "audio": True,
                "video": False, # Only audio
            },
            audio_frame_callback=audio_frame_callback,
            sendback_audio=False, # Don't send audio back to the browser client
            async_processing=True,
        )

        # Add instructions for the two-step start process
        st.markdown("**Instructions:**")
        st.markdown("1. Click the ▶️ button *inside the component box above* to activate the microphone stream.")
        st.markdown("2. Click the button below to start/stop recording the active stream.")

        # Single button for Start/Stop Recording
        if st.session_state.is_recording:
            if st.button("🛑 Stop Recording", key="record_toggle", type="primary"):
                st.session_state.is_recording = False
                st.write("DEBUG: Stop Recording button pressed.") # DEBUG
                st.rerun()
        else:
            if st.button("🎙️ Start Recording", key="record_toggle"):
                # Ensure the user has started the webrtc component stream first
                if webrtc_ctx and webrtc_ctx.state.playing:
                    st.session_state.is_recording = True
                    st.session_state.audio_buffer = queue.Queue() # Clear previous buffer
                    st.session_state.live_transcription_results = (None, None, None)
                    st.write("DEBUG: Start Recording button pressed.") # DEBUG
                    st.rerun()
                else:
                    st.warning("Please click the initial ▶️ button in the component above to start the audio stream first.")

        status_indicator = st.empty()
        if st.session_state.is_recording:
            status_indicator.success("🔴 Recording... Speak now!")
            # Display the audio level progress bar
            st.progress(st.session_state.audio_level, text="Mic Level")
        else:
            status_indicator.info("⚪ Not Recording")
            # Clear progress bar when not recording
            st.progress(0.0, text="Mic Level")

        # Process audio when recording stops and buffer has data
        # Add explicit check if the condition is met
        should_process = not st.session_state.is_recording and not st.session_state.audio_buffer.empty()
        st.write(f"DEBUG: Should Process Block Reached: {should_process}") # DEBUG log

        if should_process:
            st.write("DEBUG: Entering processing block...") # DEBUG log
            status_indicator.info("⏳ Processing recorded audio...")
            # Combine audio chunks from the queue
            audio_data = []
            while not st.session_state.audio_buffer.empty():
                audio_data.append(st.session_state.audio_buffer.get())

            st.write(f"DEBUG: Combined {len(audio_data)} audio chunks.") # DEBUG log

            if audio_data:
                full_audio_np = np.concatenate(audio_data, axis=0)
                st.write(f"DEBUG: Concatenated audio shape: {full_audio_np.shape}") # DEBUG log

                # Save NumPy array to a temporary file (WhisperX needs a file path)
                # Use the sample rate from the webrtc component (usually 48kHz, but check)
                sample_rate = 48000 # Default to 48kHz as fallback
                try:
                    # Attempt to get actual sample rate if receiver exists
                    if webrtc_ctx and webrtc_ctx.audio_receiver:
                        stats = webrtc_ctx.audio_receiver.get_stats()
                        if stats and "codec" in stats and "sampleRate" in stats["codec"]:
                            sample_rate = stats["codec"]["sampleRate"]
                            st.write(f"DEBUG: Got sample rate from webrtc: {sample_rate}") # DEBUG log
                        else:
                             st.write("DEBUG: Could not get sample rate from webrtc stats, using default.") # DEBUG log
                    else:
                         st.write("DEBUG: webrtc_ctx or audio_receiver not available, using default sample rate.") # DEBUG log
                except Exception as e:
                     st.write(f"DEBUG: Error getting sample rate: {e}, using default.") # DEBUG log


                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio_file:
                    sf.write(tmp_audio_file.name, full_audio_np, sample_rate)
                    tmp_path = tmp_audio_file.name
                    st.write(f"DEBUG: Saved temporary audio to: {tmp_path}") # DEBUG log

                # --- Add Audio Playback for Debugging ---
                st.write("**DEBUG: Playing back recorded audio...**")
                try:
                    with open(tmp_path, 'rb') as f:
                        st.audio(f.read(), format='audio/wav', sample_rate=sample_rate)
                except Exception as e:
                    st.error(f"Error playing back temp audio: {e}")
                # --- End Audio Playback ---

                # Simulate FileUploader object for process_audio_with_diarization
                class MockFileUploader:
                    def __init__(self, path):
                        self.name = os.path.basename(path)
                        self.path = path
                    def getvalue(self):
                        with open(self.path, 'rb') as f:
                            return f.read()

                mock_audio_file = MockFileUploader(tmp_path)

                # Process using the existing function
                if whisper_model and diarize_model:
                     st.write("DEBUG: Models loaded, calling process_audio_with_diarization...") # DEBUG log
                     with st.spinner("Transcribing live audio..."):
                        highlighted, clean, segments_data = process_audio_with_diarization(
                            mock_audio_file, whisper_model, diarize_model, timestamp_freq
                        )
                        st.write("DEBUG: process_audio_with_diarization finished.") # DEBUG log
                        st.session_state.live_transcription_results = (highlighted, clean, segments_data)

                        # Clean up temp file
                        try:
                             os.unlink(tmp_path)
                             st.write(f"DEBUG: Deleted temporary file: {tmp_path}") # DEBUG log
                        except Exception as e:
                             st.error(f"Error deleting temporary file {tmp_path}: {e}")

                        # Clear buffer queue after processing
                        # Move buffer clearing here to ensure it happens even if rerun fails
                        while not st.session_state.audio_buffer.empty():
                            st.session_state.audio_buffer.get()
                        st.write("DEBUG: Audio buffer cleared.") # DEBUG log

                        status_indicator.success("✅ Transcription Complete!")
                        st.write("DEBUG: Rerunning to display results...") # DEBUG log
                        # Ensure state is updated before rerun
                        st.session_state.audio_level = 0.0
                        st.rerun() # Rerun to display results
                else:
                     st.error("Models not loaded properly, cannot transcribe live audio.")
                     status_indicator.error("Error during processing.")
                     # Clear buffer queue even on error
                     while not st.session_state.audio_buffer.empty():
                         st.session_state.audio_buffer.get()
                     st.write("DEBUG: Audio buffer cleared on model load error.") # DEBUG log
                     # Clear results on error
                     st.session_state.live_transcription_results = (None, None, None)
            else:
                st.warning("No audio data recorded.")
                status_indicator.warning("No audio data recorded.") # Update status too
                # Clear buffer queue
                while not st.session_state.audio_buffer.empty():
                    st.session_state.audio_buffer.get()
                st.write("DEBUG: Audio buffer cleared because no data was recorded.") # DEBUG log
                # Clear results if no data
                st.session_state.live_transcription_results = (None, None, None)

        # Display live transcription results
        if st.session_state.live_transcription_results[0]:
            st.subheader("Live Transcription Result:")
            hl_text, cl_text, segments_data = st.session_state.live_transcription_results
            st.markdown(hl_text, unsafe_allow_html=True)
            # Optionally add download buttons for live results too
            col1_live, col2_live = st.columns(2) # Columns for live downloads
            with col1_live:
                st.download_button(
                    label="📥 Download Live Transcript (TXT)",
                    data=cl_text,
                    file_name="live_transcript.txt",
                    mime="text/plain",
                    key="download_txt_live" # Add unique key
                )
            with col2_live:
                # Add CSV download for live results
                if segments_data:
                    csv_data_live = create_csv_data(segments_data)
                    st.download_button(
                        label="📥 Download Live Transcript (CSV)",
                        data=csv_data_live,
                        file_name="live_transcript.csv",
                        mime="text/csv",
                        key="download_csv_live" # Add unique key
                    )
                else:
                    st.info("No segment data available for CSV export (Live).", icon="ℹ️")

if __name__ == "__main__":
    main() # Keep this line 