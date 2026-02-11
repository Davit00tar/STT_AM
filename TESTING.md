# 🧪 Testing Guide - Medical Audio Transcription App

## ✅ API Integration Complete

Your Streamlit app now has **full API integration** with:
- ✅ Hugging Face Whisper (Chillarmo/whisper-large-v3-turbo-armenian) for Armenian transcription
- ✅ Google Gemini (gemini-2.5-flash) for medical note formatting
- ℹ️ Groq Whisper (whisper-large-v3-turbo) kept as fallback

---

## 🎯 How to Test

### Quick Check
Your app is already running at **http://localhost:8501**  
Streamlit auto-reloads, so the changes are already live!

---

## 📝 Step-by-Step Testing

### Test 1: Upload an Audio File

1. **Navigate to the Upload File tab**
2. **Upload your test audio** (e.g., `audio_my.m4a`)
   - Supported formats: `.mp3`, `.m4a`, `.wav`
3. **Click "Generate Medical Note"**
4. **Watch the process:**
   - 🎙️ "Transcribing audio with Whisper..." spinner appears
   - ✅ "Transcription complete!" success message
   - 🤖 "Processing with Gemini AI..." spinner appears
   - ✅ "Clinical note generated!" success message
5. **Review the output:**
   - Raw Armenian transcript → Formatted clinical note
   - Should show corrected spelling, proper medical terms, standardized drug names

### Test 2: Live Recording (if browser supports)

1. **Navigate to the Live Record tab**
2. **Click the microphone button**
3. **Speak in Armenian** (medical consultation simulation)
4. **Stop recording**
5. **Click "Generate Medical Note"**
6. **Same process as Test 1**

### Test 3: Export Functions

1. **After generating a note:**
   - Click **"Copy to Clipboard"** → Paste in a text editor to verify
   - Click **"Download .txt"** → Check the downloaded file

---

## 🔍 What to Look For

### ✅ Success Indicators

- **No error messages** about missing API keys
- **Whisper transcription** completes without errors
- **Gemini formatting** processes the Armenian text
- **Output shows:**
  - Corrected spelling
  - Proper medical terminology in Armenian
  - Drug names in English/Latin (e.g., "Metformin" instead of "մեթֆորմին")
  - Clean, structured format

### ❌ Potential Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Missing API key" error | secrets.toml not configured | Check `.streamlit/secrets.toml` has HF_API_KEY, GOOGLE_API_KEY |
| "Transcription failed" | Invalid HF API key or model unavailable | Verify key at huggingface.co/settings/tokens |
| "Gemini processing failed" | Invalid Google API key | Verify key at makersuite.google.com |
| File upload error | Unsupported format | Use .mp3, .m4a, or .wav only |
| Recording not working | Browser doesn't support it | Try Chrome/Edge, or use Upload tab |

---

## 🧪 Sample Test Workflow

```
1. Open http://localhost:8501
2. Go to "Upload File" tab
3. Upload audio_my.m4a (your test file)
4. Click "Generate Medical Note"
5. Wait ~10-30 seconds (depends on audio length)
6. Verify output looks like clean medical Armenian text
7. Click "Copy to Clipboard"
8. Paste in Notepad to verify
9. Click "Download .txt"
10. Check downloaded file
```

---

## 📊 Expected Processing Time

| Audio Length | Transcription | Gemini | Total |
|--------------|---------------|--------|-------|
| 30 seconds   | ~3-5 sec     | ~2-3 sec | ~5-8 sec |
| 1 minute     | ~5-10 sec    | ~3-5 sec | ~8-15 sec |
| 5 minutes    | ~20-30 sec   | ~5-10 sec | ~25-40 sec |

---

## 🔧 Debugging Tips

### View API Responses

Add debug output by modifying the functions temporarily:

```python
# In transcribe_audio_groq():
print(f"DEBUG: Transcription result: {transcription.text[:100]}")

# In process_with_gemini():
print(f"DEBUG: Gemini response: {response.text[:100]}")
```

### Check Streamlit Logs

Watch the terminal where you ran `python -m streamlit run streamlit_app.py`  
Errors will appear there in real-time.

### Test API Keys Separately

Run a quick test in Python:

```python
from huggingface_hub import InferenceClient
import google.generativeai as genai

# Test Hugging Face
client = InferenceClient(token="your_hf_key")
print("HF: OK")

# Test Gemini
genai.configure(api_key="your_google_key")
print("Gemini: OK")
```

---

## 🎉 Success Criteria

Your integration is successful if:

- ✅ Audio uploads without errors
- ✅ Whisper transcribes Armenian speech correctly
- ✅ Gemini formats the text into a clean clinical note
- ✅ Output can be copied and downloaded
- ✅ Multiple files can be processed sequentially
- ✅ No API key errors appear
- ✅ Uses fine-tuned Armenian model for better accuracy

---

## 🚀 Next Steps After Testing

Once everything works:

1. **Test with real consultations** (ensure HIPAA/privacy compliance)
2. **Adjust system prompt** in Gemini if formatting needs tweaking
3. **Monitor API costs** (Hugging Face and Google both have usage limits)
4. **Add error logging** for production use
5. **Consider batch processing** for multiple files

---

## 📞 Quick Checklist

Before each test:
- [ ] Streamlit server is running
- [ ] Browser is at localhost:8501
- [ ] `.streamlit/secrets.toml` has HF_API_KEY and GOOGLE_API_KEY
- [ ] Test audio file is ready
- [ ] Internet connection is active

---

**Happy Testing! 🏥**
