# 🏥 Clinical Audio Transcription System

A Streamlit-based web application that converts medical audio consultations into structured Armenian clinical records using Groq's Whisper API and Google Gemini.

## 🎯 Features

- **Dual Input Methods**
  - 📁 Upload audio files (.mp3, .m4a, .wav)
  - 🎤 Live browser recording

- **AI-Powered Processing**
  - Automatic transcription using Groq's Whisper (whisper-large-v3-turbo)
  - Intelligent formatting with Google Gemini
  - Armenian language support with medical terminology

- **Professional Output**
  - Clean, structured clinical notes
  - Copy to clipboard functionality
  - Download as .txt file

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Groq API key ([Get one here](https://console.groq.com))
- Google AI API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd c:\Users\Asus\Documents\ScanMind.AI\speech_to_text\stt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up secrets**
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Add your API keys:
     ```toml
     GROQ_API_KEY = "your-groq-api-key"
     GOOGLE_API_KEY = "your-google-api-key"
     ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

## 📁 Project Structure

```
stt/
├── streamlit_app.py          # Main application
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── .streamlit/
    └── secrets.toml.example   # API key template
```

## 🔧 Configuration

### API Keys

The application uses Streamlit's secrets management for secure API key storage:

```python
# Access in code
groq_key = st.secrets["GROQ_API_KEY"]
google_key = st.secrets["GOOGLE_API_KEY"]
```

### Models

- **Transcription**: `whisper-large-v3-turbo` (Groq)
- **Formatting**: `gemini-2.5-flash` or `gemini-3-flash` (Google)

## 🔒 Security

- API keys stored in `.streamlit/secrets.toml` (gitignored)
- Temporary audio files automatically cleaned
- No data persistence beyond session
- All processing happens server-side

## 🛠️ Development

### Adding Your API Logic

The application has placeholder functions ready for integration:

1. **`transcribe_audio_groq(audio_file_path)`**
   - Location: `streamlit_app.py` (line ~60)
   - Replace with your Groq Whisper API call

2. **`process_with_gemini(raw_transcript)`**
   - Location: `streamlit_app.py` (line ~73)
   - Replace with your Gemini API call
   - System instruction already defined

### Testing

Test the UI without API calls:
```bash
streamlit run streamlit_app.py
```

The placeholders will return dummy text to verify the UI flow.

## 📝 Usage

1. **Select Input Method**
   - Choose "Upload File" tab for existing recordings
   - Choose "Live Record" tab to record directly

2. **Provide Audio**
   - Upload: Select your audio file
   - Record: Click the microphone button

3. **Generate Note**
   - Click "Generate Medical Note"
   - Wait for processing (shows loading spinners)

4. **Export Results**
   - Copy to clipboard for pasting elsewhere
   - Download as .txt file for records

## 🐛 Troubleshooting

### Audio Recording Not Working
```bash
pip install --upgrade streamlit-audiorec
```

### Streamlit Not Starting
```bash
# Clear cache
streamlit cache clear
```

### API Errors
- Verify API keys in `.streamlit/secrets.toml`
- Check internet connection
- Ensure API quotas are not exceeded

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Groq API Docs](https://console.groq.com/docs)
- [Google Gemini API](https://ai.google.dev/docs)

## 📄 License

Internal use for medical professionals.

---

**Built with ❤️ for Armenian medical community**
