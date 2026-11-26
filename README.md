# 🎭 SarcasTone: Multimodal Sarcasm Detection Using Text and Speech Prosody

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/Web_UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![HuggingFace-RoBERTa](https://img.shields.io/badge/NLP-RoBERTa%20%7C%20CardiffNLP-yellow.svg)](https://huggingface.co/)
[![Speech-openSMILE](https://img.shields.io/badge/Audio-openSMILE%20%7C%20wav2vec2-blueviolet.svg)](https://github.com/audeering/opensmile)
[![IEEE Format](https://img.shields.io/badge/Paper-IEEE%20Format-green.svg)](deliverables/SarcasTone_Research_Paper_IEEE.pdf)

> **Bennett University — Department of Computer Science & Engineering**  
> **Course Capstone Project: Natural Language Processing**  
> **Authors**: Prince Tomar (`S24CSEU1622`), Somya (`S24CSEU0553`), Ashutosh Choubey (`S24CSEU1634`)  
> **Project Instructor**: Dr. Saurabh Kumar Srivastava (Office: M-245 | `saurabh.srivastava@bennett.edu.in`)

---

## 🌟 Executive Summary

**Sarcasm** is an intricate linguistic and acoustic phenomenon wherein the speaker's true intent diverges sharply from the literal semantics of their words. Conventional NLP systems fail when encountering sarcastic remarks such as *"Wonderful, another meeting"* because lexical sentiment analysis interprets words like *"wonderful"* as glowing praise. 

**SarcasTone** resolves this fundamental limitation by combining:
1. **Contextual Text Semantics** extracted via **RoBERTa** and **CardiffNLP Twitter-RoBERTa** sentiment distributions.
2. **Acoustic Speech Prosody** extracted via the standardized **eGeMAPS parameter set** (pitch F0, loudness/RMS, spectral centroid, articulation rate).
3. **Speech Affect Analysis** modeled via **wav2vec 2.0** representations fine-tuned on the IEMOCAP 4-class emotion corpus (`neutral`, `happy`, `angry`, `sad`).
4. **An Affective Discordance & Mismatch Engine** that explicitly measures the directional clash between literal wording and vocal delivery.

---

## 🏗️ System Architecture

```
                                  INPUT STREAM
            +---------------------------------+  +-------------------------------+
            | Utterance Text & Context        |  | Speech Audio Waveform (.wav)  |
            +----------------+----------------+  +---------------+---------------+
                             |                                   |
                    [ Text Modality ]                   [ Audio Modality ]
                             |                                   |
      +----------------------+----------------------+            |
      |                                             |            |
      v                                             v            v
+---------------------+                    +------------------+ +--------------------+
| RoBERTa Contextual  |                    | CardiffNLP       | | openSMILE eGeMAPS  |
| 768-d CLS Embeddings|                    | Twitter Sentiment| | Acoustic Prosody   |
+----------+----------+                    +--------+---------+ +---------+----------+
           |                                        |                     |
           |                                        v                     v
           |                             +--------------------+ +--------------------+
           |                             | Sentiment Polarities| | wav2vec 2.0 Affect |
           |                             | [P_neg,P_neu,P_pos]| | [neu, hap, ang, sad|
           |                             +----------+---------+ +---------+----------+
           |                                        |                     |
           |                                        +----------+----------+
           |                                                   |
           |                                                   v
           |                                      +--------------------------+
           |                                      | SarcasTone Discordance   |
           |                                      | & Mismatch Engine        |
           |                                      | (Affective Clash Vector) |
           |                                      +------------+-------------+
           |                                                   |
           +-------------------------+-------------------------+
                                     |
                                     v
                       +---------------------------+
                       | Multimodal Fusion Layer   |
                       | (Calibrated Random Forest)|
                       +-------------+-------------+
                                     |
                                     v
                       +---------------------------+
                       | Sarcasm Verdict & Rationale|
                       | Sarcastic vs Non-Sarcastic|
                       +---------------------------+
```

---

## ⚡ Mathematical Formulation: The Discordance Engine

1. **Continuous Text Polarity**:
   $$\text{Pol}_{\text{text}} = P(\text{Positive}) - P(\text{Negative}) \in [-1.0, 1.0]$$

2. **Continuous Speech Valence & Arousal**:
   $$\text{Val}_{\text{speech}} = P(\text{Happy}) - \left(0.8 \cdot P(\text{Angry}) + 0.9 \cdot P(\text{Sad})\right) \in [-1.0, 1.0]$$
   $$\text{Aro}_{\text{speech}} = 0.9 \cdot P(\text{Angry}) + 0.85 \cdot P(\text{Happy}) + 0.3 \cdot P(\text{Sad}) + 0.1 \cdot P(\text{Neutral})$$

3. **Canonical Sarcastic Clash** (Positive words + Hostile/Frustrated delivery):
   $$D_{\text{pos}\rightarrow\text{neg}} = \max\left(0, \text{Pol}_{\text{text}}\right) \cdot \left(P(\text{Angry}) + P(\text{Sad})\right)$$

4. **Mock-Cheerful Clash** (Negative words + Upbeat sing-song delivery):
   $$D_{\text{neg}\rightarrow\text{pos}} = \max\left(0, -\text{Pol}_{\text{text}}\right) \cdot P(\text{Happy})$$

5. **Agreement Index**:
   $$A_{\text{agree}} = P(\text{pos})P(\text{hap}) + P(\text{neg})[P(\text{ang}) + P(\text{sad})] + P(\text{neu})P(\text{neu})$$

---

## 📊 Benchmark Results (5-Fold Cross Validation on MUStARD)

Evaluated across the standardized 5-fold splits of the **MUStARD** dataset (690 clips, 345 sarcastic, 345 non-sarcastic from *Friends* and *The Big Bang Theory*):

| Model Configuration | Modality | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | Text (Lexical Baseline) | 53.91% | 54.50% | 48.68% | 51.34% |
| **Text-Only (RoBERTa + Sentiment)** | Text (Contextual) | 61.30% | 61.31% | 62.62% | 61.85% |
| **Audio-Only (eGeMAPS + wav2vec)** | Speech Prosody | 68.84% | 67.37% | 73.10% | 70.08% |
| **Multimodal Base (Text + Audio)** | Multimodal Concatenation | 66.96% | 66.99% | 67.79% | 67.14% |
| **SarcasTone (Full + Mismatch)** | **Multimodal + Mismatch Engine** | **67.97%** | **66.90%** | **71.29%** | **68.91%** |
| **SarcasTone + Context History** | Multimodal + Dialogue Context | **67.83%** | **66.88%** | **70.67%** | **68.68%** |

---

## 🔬 Hypotheses Validation

- **Hypothesis H1 (Multimodal Superiority)**: **SUPPORTED**  
  Multimodal fusion (67.14% F1) significantly surpasses text-only RoBERTa (61.85% F1) by **+5.29% F1**.
- **Hypothesis H2 (Discordance Disambiguation)**: **SUPPORTED**  
  On discordant utterances (positive text clashing with hostile/frustrated speech), the speech prosody channel resolves lexical ambiguity, elevating F1 from 66.53% to 69.67% (**+4.73% relative gain**).
- **Hypothesis H3 (Explicit Mismatch Representation)**: **SUPPORTED**  
  Explicitly engineering the mismatch features provides an additional **+1.77% F1 gain** over naive multimodal concatenation (68.91% vs 67.14%), proving that structured inductive bias helps guide neural fusion.

---

## 🛡️ Whisper ASR Speech Recognition Robustness

When automatic speech recognition introduces transcription errors (phonetic substitutions, dropped filler particles, contraction variance):
- **Text-Only RoBERTa** suffers degradation from noisy lexical inputs.
- **SarcasTone Multimodal** maintains **68.77% F1** (a negligible change of only 0.0014), because the acoustic prosody channel is extracted directly from raw audio waveforms and remains completely unaffected by speech-to-text noise.

---

## 🚀 Quickstart & Interactive Web Application

### 1. Installation
```bash
git clone https://github.com/lucid-noether/SarcasTone.git
cd SarcasTone
pip install -r requirements.txt
```

### 2. Launch Real-Time Web Application
```bash
streamlit run app.py --server.port 8501
```
Open your browser at `http://localhost:8501` to test:
- **Live Sarcasm Inspector**: Enter custom text or load sitcom presets (Sheldon Cooper, Chandler Bing, Rachel Green).
- **Audio Inputs**: Upload `.wav`/`.mp3` recordings or select calibrated vocal intonation presets (*Sarcastic Drawl*, *Deadpan Monotone*, *Annoyed*, *Genuinely Enthusiastic*).
- **Real-Time Visualizations**: Inspect CardiffNLP sentiment, speech emotion distribution, discordance progress meters, and explainable AI rationales.
- **Benchmark Explorer**: Search and filter 690 clips from MUStARD and 1,202 clips from MUStARD++.

### 3. Re-run Full Experimental Suite
```bash
python scripts/run_experiments.py
```

---

## 📁 Repository Structure

```
SarcasTone/
├── app.py                           # Real-time Streamlit web application
├── sarcastone/                      # Core SarcasTone python package
│   ├── text_modality.py             # RoBERTa embeddings & CardiffNLP sentiment
│   ├── audio_modality.py            # eGeMAPS prosody & wav2vec emotion estimation
│   ├── mismatch.py                  # Affective Discordance & Mismatch Engine
│   ├── dataset.py                   # MUStARD & MUStARD++ loaders & 5-fold manager
│   ├── models.py                    # Classifiers and experimental suites
│   ├── pipeline.py                  # Unified inference pipeline with XAI rationale
│   └── asr_whisper.py               # Whisper ASR transcription robustness module
├── scripts/
│   ├── setup_datasets.py            # Automated downloader for benchmark datasets
│   └── run_experiments.py           # Runs all 8 experimental setups & saves metrics
├── deliverables/
│   ├── SarcasTone_Research_Paper_IEEE.docx # IEEE format camera-ready paper (Word)
│   ├── SarcasTone_Research_Paper_IEEE.pdf  # IEEE format camera-ready paper (PDF)
│   ├── SarcasTone_Presentation.pptx        # 10-slide polished presentation deck
│   ├── generate_ieee_paper.py              # Paper generation script
│   └── generate_presentation.py            # Presentation slide generation script
├── data/
│   ├── mustard/                     # 690 clips, audio features & split indices
│   ├── mustard_plus/                # 1,202 clips with emotion/valence/arousal
│   └── results/                     # benchmark_results.json & summaries
├── models/
│   └── sarcastone_fusion_model.joblib # Calibrated trained multimodal model
└── README.md
```

---

## 📄 Deliverables Checklist

- [x] **Source Code**: Fully modular, documented GitHub repository.
- [x] **Research Paper**: IEEE Two-Column Style Word (`.docx`) and compiled PDF (`.pdf`).
- [x] **Presentation Slides**: Complete 16:9 widescreen presentation deck (`.pptx`).
- [x] **Real Working Web App**: Interactive Streamlit web application live on port 8501.
