# 🎭 SarcasTone: Multimodal Sarcasm Detection Using Text and Speech Prosody

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/Web_UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![HuggingFace-RoBERTa](https://img.shields.io/badge/NLP-RoBERTa%20%7C%20CardiffNLP-yellow.svg)](https://huggingface.co/)
[![Speech-Prosody](https://img.shields.io/badge/Audio-Librosa%20%7C%20eGeMAPS-blueviolet.svg)](https://librosa.org/)
[![IEEE Format](https://img.shields.io/badge/Paper-IEEE%20Format-green.svg)](deliverables/SarcasTone_Research_Paper_IEEE.pdf)

> **Bennett University — Department of Computer Science & Engineering**  
> **Course Capstone Project: Natural Language Processing**  
> **Authors**: Prince Tomar (`S24CSEU1622`), Somya (`S24CSEU0553`), Ashutosh Choubey (`S24CSEU1634`)  
> **Project Supervisor**: Dr. Saurabh Kumar Srivastava (Office: M-245 | `saurabh.srivastava@bennett.edu.in`)

---

## 🌟 Executive Summary

**Sarcasm** is an intricate linguistic and acoustic phenomenon wherein the speaker's true communicative intent diverges sharply from the literal semantics of their words. Conventional NLP systems fail when encountering sarcastic remarks such as *"Wonderful, another meeting"* because lexical sentiment analysis interprets words like *"wonderful"* as positive praise.

**SarcasTone** resolves this fundamental limitation by combining:
1. **Contextual Text Semantics** extracted via **RoBERTa** and **CardiffNLP Twitter-RoBERTa** sentiment distributions (768-d CLS + 3-class sentiment probabilities).
2. **Acoustic Speech Prosody** extracted via **librosa** and **eGeMAPS-inspired features** (fundamental frequency F0, loudness/RMS dynamics, zero-crossing rate, speech tempo, vocal jitter).
3. **Speech Affect Analysis** modeling acoustic emotion across 4 classes (`neutral`, `happy`, `angry/annoyed`, `sad/frustrated`).
4. **An Affective Discordance & Mismatch Engine** that explicitly measures the directional clash between literal wording and vocal delivery.
5. **Calibrated Machine Learning Classifiers** (MLP, Gradient Boosting, Random Forest) trained and evaluated on the benchmark **MUStARD** dataset with 5-fold cross-validation.

---

## 🚀 Key Features: Try SarcasTone – Multimodal Sarcasm Detector

SarcasTone features a real-time web application (`app.py`) providing three distinct prediction modes:

### 📖 Mode A: Text Sarcasm Detection
- Enter any custom sentence or utterance with optional conversational context.
- Analyzes contextual embeddings and sentiment polarity using RoBERTa.
- Returns real prediction (Sarcastic vs. Non-Sarcastic), confidence percentage, and sentiment distribution breakdown.

### 🎙️ Mode B: Audio Sarcasm Detection
- **Microphone Recording:** Record your voice directly in the browser via native microphone input.
- **Audio File Upload:** Upload `.wav`, `.mp3`, or `.ogg` audio files.
- **Vocal Presets:** Choose from calibrated SITCOM acoustic profiles (*Sarcastic Drawl*, *Deadpan Monotone*, *Annoyed*, *Genuinely Enthusiastic*).
- Extracts pitch contour, RMS energy, and acoustic affect to predict sarcastic intent.

### ⚡ Mode C: Multimodal Detection (Text + Audio Fusion)
- Submit both text and spoken audio together.
- Processes linguistic and acoustic channels through dedicated feature extraction pipelines.
- Evaluates the **affective discordance** (e.g., positive wording delivered with irritated intonation).
- Generates a **3-way comparative scorecard**: Text Model prediction, Audio Model prediction, and Final Fused prediction.

---

## 📊 Dataset Audit Summary

A rigorous audit of the repository datasets was conducted:
- Detailed audit documentation: [`DATASET_AUDIT.md`](DATASET_AUDIT.md)

| Requirement | MUStARD | MUStARD++ |
|---|---|---|
| **Text available** | ✅ Yes – 690 utterances in `sarcasm_data.json` | ✅ Yes – 1,202 labeled sentences in `mustard++_text.csv` |
| **Audio available** | ⚠️ 690 pre-extracted feature matrices (`audio_features.p`) | ❌ No audio files present |
| **Video available** | ❌ None | ❌ None |
| **Total samples** | 690 (345 sarcastic, 345 non-sarcastic) | 1,202 labeled (601 sarcastic, 601 non-sarcastic) |
| **Text-Audio alignment** | ✅ Perfect 690/690 key match | ❌ N/A (text only) |
| **Dataset status** | **Partially Multimodal** (aligned acoustic matrices) | **Text-Only** |

---

## 📈 Model Evaluation (5-Fold Cross Validation on MUStARD)

All models were evaluated on the 690 MUStARD benchmark samples across 5 stratified folds. No hardcoded metrics:
- Detailed evaluation report: [`MODEL_EVALUATION.md`](MODEL_EVALUATION.md)

| Model Configuration | Modality | Classifier | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Text-Only** | RoBERTa + Sentiment (771-d) | MLP | 62.17% | 62.13% | 63.49% | **62.67%** |
| **Audio-Only** | Prosody + Emotion (59-d) | Gradient Boosting | 67.68% | 66.97% | 69.87% | **68.30%** |
| **Multimodal Fusion** | Text + Audio + Mismatch (841-d) | MLP | 66.81% | 65.33% | 72.19% | **68.40%** |
| **Text + Audio Base** | Text + Audio Concatenation (830-d) | Random Forest | 68.70% | 68.55% | 69.89% | **69.07%** |

---

## 🌐 Production REST API Endpoints (`api/index.py`)

A serverless-ready HTTP API is provided in `api/index.py`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check, version info, and model readiness status |
| `GET` | `/api/metrics` | 5-fold cross-validation performance metrics across all models |
| `GET` | `/api/dataset` | Dataset audit counts and availability summary |
| `POST` | `/api/predict/text` | Text sarcasm prediction (`{"utterance": "...", "context": [...]}`) |
| `POST` | `/api/predict/audio` | Audio sarcasm prediction (`{"preset_name": "..."}` or `{"audio_base64": "..."}`) |
| `POST` | `/api/predict/multimodal` | Full multimodal fusion prediction (`{"utterance": "...", "preset_name": "..."}`) |

---

## 💻 Quickstart & Instructions to Run

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- PyTorch, Transformers, Scikit-learn, Librosa, Soundfile, Streamlit

### 2. Setup
```bash
git clone https://github.com/achoubey427/SarcasTone-Multimodal-Sarcasm-Detection.git
cd SarcasTone-Multimodal-Sarcasm-Detection
pip install -r requirements.txt
```

### 3. Launch Web Application
```bash
streamlit run app.py --server.port 8501
```
Open `http://localhost:8501` to use the interactive detector.

### 4. Run Automated End-to-End Test Suite
```bash
python tests/test_pipeline_and_api.py
```

### 5. Re-train Models
```bash
python scripts/train_models.py
```

---

## 📁 Repository Structure

```
SarcasTone-Multimodal-Sarcasm-Detection/
├── app.py                             # Interactive Streamlit Web Application (Modes A, B, C)
├── DATASET_AUDIT.md                   # Complete audit report for MUStARD & MUStARD++
├── MODEL_EVALUATION.md                # 5-fold cross-validation evaluation report
├── api/
│   ├── index.py                       # REST API handler for text, audio & multimodal endpoints
│   └── requirements.txt               # API dependencies
├── sarcastone/                        # Core Python package
│   ├── text_modality.py               # RoBERTa embeddings & CardiffNLP sentiment
│   ├── audio_modality.py              # Prosodic feature extraction & affect estimation
│   ├── mismatch.py                    # Affective Discordance & Mismatch Engine
│   ├── dataset.py                     # MUStARD & MUStARD++ dataset managers
│   ├── models.py                      # Model building & experimental suites
│   ├── pipeline.py                    # Unified inference pipeline for text, audio & fusion
│   └── asr_whisper.py                 # Whisper ASR transcription robustness simulation
├── models/
│   ├── sarcastone_text_model.joblib   # Trained text-only classifier
│   ├── sarcastone_audio_model.joblib  # Trained audio-only classifier
│   └── sarcastone_multimodal_model.joblib # Trained multimodal fusion classifier
├── scripts/
│   ├── audit_datasets.py              # Automated dataset verification script
│   ├── train_models.py                # 5-fold CV training pipeline for all 3 models
│   └── run_experiments.py             # Baseline benchmark suite
├── tests/
│   └── test_pipeline_and_api.py       # Comprehensive end-to-end test suite
├── deliverables/
│   ├── SarcasTone_Research_Paper_IEEE.pdf  # Camera-ready IEEE format paper
│   ├── SarcasTone_Faculty_Project_Report.pdf # Faculty-ready project report
│   └── SarcasTone_Presentation.pptx   # 10-slide presentation deck
└── README.md
```

---

## 📄 Academic Citation & Project Context

Developed as an undergraduate Capstone Project at **Bennett University (Times of India Group)**, Greater Noida, India.
- Supervisor: Dr. Saurabh Kumar Srivastava
- Students: Prince Tomar, Somya, Ashutosh Choubey
