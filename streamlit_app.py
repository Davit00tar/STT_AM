"""
Medical Audio to Clinical Notes - Streamlit Application
Converts audio consultations into structured Armenian clinical records
using Hugging Face (fine-tuned Whisper) and Google Gemini.
"""

import streamlit as st
import tempfile
import os
import time
from datetime import datetime
from pathlib import Path
from groq import Groq
from huggingface_hub import InferenceClient
from gradio_client import Client as GradioClient, handle_file
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="Speech to Text for 🇦🇲Armenian Medical Records",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with Armenian font support and modern medical aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Armenian:wght@400;500;600;700&display=swap');
    
    .main {
        background-color: #fafbfc;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        font-family: 'Noto Sans Armenian', 'Segoe UI', Arial, sans-serif;
    }
    h1 {
        color: #1a237e;
        font-family: 'Noto Sans Armenian', sans-serif;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    h3 {
        color: #283593;
        font-weight: 600;
        font-family: 'Noto Sans Armenian', sans-serif;
    }
    p, div, span {
        font-family: 'Noto Sans Armenian', 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0277bd 0%, #01579b 100%);
        color: white;
        font-family: 'Noto Sans Armenian', sans-serif;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.85rem 2.5rem;
        border: none;
        font-size: 1.15rem;
        width: 100%;
        margin-top: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(1, 87, 155, 0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #01579b 0%, #0277bd 100%);
        box-shadow: 0 6px 16px rgba(1, 87, 155, 0.3);
        transform: translateY(-2px);
    }
    .stTextArea textarea {
        font-family: 'Noto Sans Armenian', 'Courier New', monospace;
        font-size: 1rem;
        border: 2px solid #b0bec5;
        border-radius: 12px;
        padding: 1rem;
        line-height: 1.6;
        transition: border-color 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #0277bd;
        box-shadow: 0 0 0 3px rgba(2, 119, 189, 0.1);
    }
    .stTabs {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Noto Sans Armenian', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.75rem 1.5rem;
        border-radius: 8px 8px 0 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize API clients using Streamlit secrets
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])  # Kept as fallback
    hf_client = InferenceClient(token=st.secrets["HF_API_KEY"])
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except KeyError as e:
    st.error(f"⚠️ Missing API key: {e}. Please configure your .streamlit/secrets.toml file.")
    st.stop()

# Initialize session state with history list
if 'transcription_history' not in st.session_state:
    st.session_state.transcription_history = []  # List of {timestamp, raw_transcript, clinical_note, source}
if 'audio_source' not in st.session_state:
    st.session_state.audio_source = None
if 'pending_audio_bytes' not in st.session_state:
    st.session_state.pending_audio_bytes = None  # Store audio bytes for processing
if 'pending_audio_name' not in st.session_state:
    st.session_state.pending_audio_name = None


def transcribe_audio_groq(audio_file_path):
    """
    Transcribe audio using Groq's Whisper API
    Args:
        audio_file_path: Path to audio file
    Returns:
        str: Transcribed text in Armenian
    """
    try:
        # Check file size
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:  # Less than 1KB is likely empty/corrupted
            raise Exception("EMPTY_AUDIO")
        
        with open(audio_file_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), file.read()),
                model="whisper-large-v3-turbo",
                language="hy",  # Armenian language code
                temperature=0,
                response_format="verbose_json",
            )
            
            # Check if transcription is empty
            if not transcription.text or len(transcription.text.strip()) < 3:
                raise Exception("EMPTY_TRANSCRIPT")
                
            return transcription.text
    except Exception as e:
        if "EMPTY" in str(e):
            raise Exception("EMPTY_AUDIO")
        raise Exception(f"Transcription error: {str(e)}")


def transcribe_audio_hf(audio_file_path):
    """
    Transcribe audio using Hugging Face Inference API with fine-tuned Armenian Whisper model
    (Chillarmo/whisper-large-v3-turbo-armenian)
    Args:
        audio_file_path: Path to audio file
    Returns:
        str: Transcribed text in Armenian
    """
    try:
        # Check file size
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:  # Less than 1KB is likely empty/corrupted
            raise Exception("EMPTY_AUDIO")
        
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()
        
        result = hf_client.automatic_speech_recognition(
            audio=audio_bytes,
            model="Chillarmo/whisper-large-v3-turbo-armenian",
        )
        
        # Extract transcription text
        transcription_text = result.text if hasattr(result, 'text') else str(result)
        
        # Check if transcription is empty
        if not transcription_text or len(transcription_text.strip()) < 3:
            raise Exception("EMPTY_TRANSCRIPT")
        
        return transcription_text
    except Exception as e:
        print(f"DEBUG HF ERROR: {type(e).__name__}: {str(e)}")  # Debug logging
        if "EMPTY" in str(e):
            raise Exception("EMPTY_AUDIO")
        raise Exception(f"Transcription error: {str(e)}")


def transcribe_audio_hf_space(audio_file_path):
    """
    Transcribe audio using the Hugging Face Space (davtar10/whisper-am-server)
    via Gradio Client. This calls a remote ZeroGPU-backed Space.
    Args:
        audio_file_path: Path to audio file
    Returns:
        str: Transcribed text in Armenian
    """
    try:
        # Check file size
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:  # Less than 1KB is likely empty/corrupted
            raise Exception("EMPTY_AUDIO")

        # Get space name from secrets, fallback to default
        space_name = st.secrets.get("HF_SPACE_NAME", "davtar10/whisper-am-server")
        hf_token = st.secrets["HF_API_KEY"]

        # Connect to the HF Space with authentication
        client = GradioClient(space_name, token=hf_token)

        # Send file to the Space for transcription
        result = client.predict(
            audio=handle_file(audio_file_path),
            api_name="/transcribe"
        )

        # Extract transcription text
        transcription_text = result if isinstance(result, str) else str(result)

        # Check if transcription is empty
        if not transcription_text or len(transcription_text.strip()) < 3:
            raise Exception("EMPTY_TRANSCRIPT")

        return transcription_text
    except Exception as e:
        print(f"DEBUG HF SPACE ERROR: {type(e).__name__}: {str(e)}")
        if "EMPTY" in str(e):
            raise Exception("EMPTY_AUDIO")
        raise Exception(f"Transcription error: {str(e)}")


def process_with_gemini(raw_transcript):
    """
    Process transcript with Google Gemini for medical formatting
    Args:
        raw_transcript: Raw Armenian transcript
    Returns:
        str: Cleaned and formatted clinical note
    """
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction="""
            **Role:** You are an expert Medical Scribe and Editor. Your task is to process a raw audio transcript of a doctor's speech and convert it into a professional, structured clinical record.

### **Strict Operational Guidelines:**

1. **Filtering & Noise Reduction:**
   * **Remove Greetings/Fillers:** Automatically filter out opening greetings such as "Բարև ձեզ" (Hello), "Ողջույն" (Greetings), or non-clinical filler phrases at the start of the recording. 
   * **Focus on Clinical Content:** Start the record directly with the medical observations, patient data, or clinical findings.

2. **Total Coverage (No Omissions):**
   * Process the transcript from the very beginning to the very end. 
   * Ensure that every piece of clinical information, symptom, diagnosis, or instruction mentioned by the doctor is captured. Do not skip any sections.

3. **Linguistic & Medical Standards:**
   * **Spelling & Grammar:** Correct all objective spelling errors and typos in the Armenian text.
   * **Medical Terminology:** Ensure all medical terms are accurate and used in their standard, formal medical Armenian form.
   * **Medication Names:** Standardize all drug names to their correct English or Latin spelling (e.g., *Ceftriaxone*, *Aspirin*).

4. **Formatting:** * Structure the output into a clean, readable medical note in a narrative format.

5. **The "Zero Additions" Ironclad Rule:**
   * **No Fabrications:** STRICTLY do not add any information, assumptions, or content that is not explicitly present in the source text.
   * **No Medical Advice:** Do not provide your own medical advice, conclusions, or "logical next steps." 
   * **No Bridging:** Do not add introductory or concluding sentences like "The patient presented with..." or "In summary..." unless the doctor specifically said those words.
   * **Your Goal:** You are a clean mirror of the provided audio. Your job is exclusively to clean and format the existing input.

### **Output Format:**
* **Language:** Formal Medical Armenian.
* **Drug Names:** English/Latin.
* **Structure:** Narrative Clinical Record.
           """
        )
        
        response = model.generate_content(raw_transcript)
        
        # Check if response has valid content
        if not response.candidates or not response.candidates[0].content.parts:
            raise Exception("EMPTY_RESPONSE")
            
        return response.text
    except Exception as e:
        if "EMPTY_RESPONSE" in str(e) or "finish_reason" in str(e):
            raise Exception("EMPTY_RESPONSE")
        raise Exception(f"Processing error: {str(e)}")


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return path"""
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None


def process_audio_pipeline(audio_file_path):
    """
    Complete processing pipeline from audio to clinical note
    Returns tuple: (clinical_note, processing_time_seconds, raw_transcript)
    """
    start_time = time.time()
    
    with st.spinner("⏳ Մշակdelays processing..."):
        # Transcription — using HF Space (Gradio Client)
        # To switch backends, uncomment ONE of the lines below and comment the active one:
        # raw_transcript = transcribe_audio_groq(audio_file_path)    # Option 1: Groq Whisper
        # raw_transcript = transcribe_audio_hf(audio_file_path)      # Option 2: HF Inference API
        raw_transcript = transcribe_audio_hf_space(audio_file_path)   # Option 3: HF Space (active)
        
        # Show raw transcript for debugging
        with st.expander("🔍 Դիտել չմշակված տեքստը (դեբագ)", expanded=False):
            st.text(raw_transcript)
        
        # Formatting - only if we have valid Armenian text
        clinical_note = process_with_gemini(raw_transcript)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return clinical_note, processing_time


# Header
st.title("Speech to Text for Armenian Medical Records")
st.markdown("Ձայնագրությունները վերածեք կառուցված հայերեն բժշկական գրառումների")

# Divider
st.divider()

# Input Section
st.subheader("📥 Ձայնային նյութ")

# Tabbed interface for audio input
tab1, tab2 = st.tabs(["📁 Բեռնել Ֆայլ", "🎤 Ձայնագրել"])

with tab1:
    st.markdown("Բեռնեք ձայնային ֆայլ (.mp3, .m4a, .wav)")
    uploaded_file = st.file_uploader(
        "Ընտրեք ձայնային ֆայլ",
        type=["mp3", "m4a", "wav"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[-1]}")
        st.session_state.audio_source = "upload"
        st.markdown(f"**Ֆայլ:** {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

with tab2:
    st.markdown("### 🎤 Ձայնագրում")
    
    audio_bytes = None  # Initialize to avoid scope issues
    
    # Instructions with troubleshooting
    st.info("""🎤 **Ինչպես ձայնագրել:**
    1. Սեղմեք ներքևի բարձրախոսի կոճակը
    2. Թույլատրեք browser-ի մուտքը (կհայտնվի թույլտվության պատուհան)
    3. Խոսեք հստակ
    4. Ավարտելուց հետո սեղմեք կանգառի կոճակը
    
    ⚠️ **Եթե բարձրախոսը չի աշխատում՝** Ստուգեք զննարկիչի կարգավորումները → Կայքի թույլտվություններ → Բարձրախոս""")
    
    # Use Streamlit's native audio_input
    try:
        audio_value = st.audio_input("Record your voice", label_visibility="visible")
        
        if audio_value is not None:
            # Read audio bytes once and store using getvalue()
            audio_bytes = audio_value.getvalue()
            audio_size_kb = len(audio_bytes) / 1024
            
            st.success("✅ **The recording was successfully saved!**")
            st.audio(audio_bytes, format="audio/wav")
            st.session_state.audio_source = "recording"
            
            # Show file info and download button
            col_info, col_download = st.columns([2, 1])
            with col_info:
                st.caption(f"📁 File size: {audio_size_kb:.1f} KB")
            with col_download:
                st.download_button(
                    label="⬇️ Download recording",
                    data=audio_bytes,
                    file_name="recording.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
            # Save recording to session state for processing
            st.session_state.pending_audio_bytes = audio_bytes
            st.session_state.pending_audio_name = f"Recording_{datetime.now().strftime('%H%M%S')}"
        else:
            st.caption("👆 Press the microphone to start recording:")
            
    except Exception as e:
        # Fallback to st_audiorec if native doesn't work
        try:
            from st_audiorec import st_audiorec
            
            st.warning("Օգտագործում է այլընտրանքային ձայնագրիչ. Կլիկ անեք բարձրափոսի կոչակի վրա:")
            audio_bytes = st_audiorec()
            
            if audio_bytes:
                audio_size_kb = len(audio_bytes) / 1024
                st.success("✅ **Ձայնագրությունը հաջողությամբ պահվեց!**")
                st.audio(audio_bytes, format="audio/wav")
                st.session_state.audio_source = "recording"
                st.caption(f"📁 Ֆայլի չափ: {audio_size_kb:.1f} KB")
                
                # Save recording
                temp_audio_path = os.path.join(tempfile.gettempdir(), "recorded_audio.wav")
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_bytes)
                    
        except ImportError:
            st.error("⚠️ Ձայնային ձայնագրություն հասանելի չէ: Խնդրում ենք օգտագործեք 'Բեռնել Ֆայլ' բաժինը:")

# Action Button
st.divider()

generate_button = st.button("🔄 Ստեղծել Բժշկական Գրառում", type="primary", use_container_width=True)

# Processing logic
if generate_button:
    audio_file_path = None
    source_name = None
    
    # Determine audio source - use session state for recording
    if st.session_state.audio_source == "upload" and uploaded_file:
        audio_file_path = save_uploaded_file(uploaded_file)
        source_name = uploaded_file.name
    elif st.session_state.audio_source == "recording" and st.session_state.pending_audio_bytes:
        # Save pending audio bytes to unique temp file
        unique_filename = f"recording_{datetime.now().strftime('%H%M%S%f')}.wav"
        audio_file_path = os.path.join(tempfile.gettempdir(), unique_filename)
        with open(audio_file_path, "wb") as f:
            f.write(st.session_state.pending_audio_bytes)
        source_name = st.session_state.pending_audio_name or "Recording"
    else:
        st.error("⚠️ Խdelays please upload an audio file or record before continuing!")
    
    # Process if we have audio
    if audio_file_path and os.path.exists(audio_file_path):
        with st.status("📤 Audio sent! Processing your recording...", expanded=True) as status:
            try:
                result, proc_time = process_audio_pipeline(audio_file_path)
                
                # Add to history (prepend so newest is first)
                record_id = datetime.now().strftime('%H%M%S%f')  # Unique ID
                new_record = {
                    'id': record_id,
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'source': source_name,
                    'clinical_note': result,
                    'processing_time': proc_time
                }
                st.session_state.transcription_history.insert(0, new_record)
                
                # Clear audio state to prevent re-processing
                st.session_state.audio_source = None
                st.session_state.pending_audio_bytes = None
                st.session_state.pending_audio_name = None
                
                # Clean up temp file
                try:
                    os.remove(audio_file_path)
                except:
                    pass
                
                status.update(label=f"✅ Completed in {proc_time:.1f} seconds!", state="complete", expanded=False)
                st.rerun()  # Refresh to show new record
            except Exception as e:
                error_msg = str(e)
                if "EMPTY_AUDIO" in error_msg or "EMPTY_TRANSCRIPT" in error_msg:
                    status.update(label="⚠️ Delays speech not detected", state="error", expanded=False)
                    st.error("⚠️ Delays speech not detected in the audio. Please try again with a longer or clearer recording:")
                elif "EMPTY_RESPONSE" in error_msg:
                    status.update(label="⚠️ Delays audio unclear", state="error", expanded=False)
                    st.error("⚠️ Delays the audio appears to be empty or unclear. Please record again with clear speech:")
                else:
                    status.update(label="❌ Processing failed", state="error", expanded=False)
                    st.error(f"❌ Processing error: {error_msg}")

# Output Section - History Based
st.divider()
st.subheader("📄 Արդյունքներ")

if st.session_state.transcription_history:
    # Clear history button
    col_header, col_clear = st.columns([4, 1])
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.transcription_history = []
            st.rerun()
    
    # Display each record as an expander
    for i, record in enumerate(st.session_state.transcription_history):
        expander_label = f"📄 {record['timestamp']} - {record['source']} ({record['processing_time']:.1f}s)"
        
        with st.expander(expander_label, expanded=(i == 0)):  # First one expanded by default
            # Editable text area
            edited_text = st.text_area(
                "Բժշկական գրառում",
                value=record['clinical_note'],
                height=250,
                label_visibility="collapsed",
                key=f"record_{record.get('id', record['timestamp'])}"
            )
            
            # Update record if edited
            if edited_text != record['clinical_note']:
                st.session_state.transcription_history[i]['clinical_note'] = edited_text
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 պատճենել", key=f"copy_{record.get('id', record['timestamp'])}", use_container_width=True):
                    st.code(edited_text)
                    st.success("✅ Delays text copied! (Use Ctrl+C)")
            with col2:
                st.download_button(
                    label="💾 ներբեռնել .txt",
                    data=edited_text,
                    file_name=f"clinical_note_{record['timestamp'].replace(':', '-')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=f"download_{record.get('id', record['timestamp'])}"
                )
else:
    st.info("👆 Բեռնեք կամ ձայնագրեք ձայնը, և սեղմեք «Ստեղծել բժշկական գրառում»՝ սկսելու համար։")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #546e7a; font-size: 0.9rem; padding: 1.5rem; font-family: \"Noto Sans Armenian\", sans-serif;'>
        🔒 Անվտանգ • 🇦🇲 Հայերեն Լեզվի Աջակցություն • 🏥 Բժշկական Մակարդակ
    </div>
""", unsafe_allow_html=True)
