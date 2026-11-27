# SarcasTone – Dataset Audit Report

> **Date:** 2026-11-27
> **Audited by:** Automated Audit Script + Manual Inspection
> **Project:** SarcasTone – Multimodal Sarcasm Detection
> **Authors:** Prince Tomar (S24CSEU1622), Somya (S24CSEU0553), Ashutosh Choubey (S24CSEU1634)
> **Supervisor:** Dr. Saurabh Kumar Srivastava, Bennett University

---

## 1. Dataset Comparison Table

| Requirement | MUStARD | MUStARD++ |
|---|---|---|
| **Text available** | ✅ Yes – 690 utterances in `sarcasm_data.json` | ✅ Yes – 1,202 labeled sentences in `mustard++_text.csv` |
| **Audio available** | ⚠️ Pre-extracted acoustic feature matrices only (`audio_features.p`) – **NO raw .wav/.mp3 files** | ❌ No – text only, no audio files or features |
| **Video available** | ❌ No video files present | ❌ No video files present |
| **Total samples** | 690 | 1,202 (of 6,041 total rows; only 1,202 have sarcasm labels) |
| **Sarcastic samples** | 345 | 601 |
| **Non-sarcastic samples** | 345 | 601 |
| **Usable audio files** | 0 raw audio files; 690 pre-extracted feature matrices (shape 283×11 to 283×18) | 0 |
| **Missing audio files** | All 690 raw audio/video clips are absent; only pre-computed acoustic features remain | All 1,202 raw audio/video clips are absent |
| **Text-audio alignment** | ✅ Perfect – all 690 JSON keys match all 690 audio_features.p keys | ❌ N/A – no audio data |
| **Dataset status** | **Partially Multimodal** – text + pre-extracted audio features | **Text-Only** |

---

## 2. Detailed Findings

### 2.1 MUStARD Dataset

**Location:** `data/mustard/`

**Files present (3 total):**
| File | Size | Description |
|---|---|---|
| `sarcasm_data.json` | JSON | 690 items with fields: `utterance`, `speaker`, `context`, `context_speakers`, `show`, `sarcasm` |
| `audio_features.p` | Pickle | 690 pre-extracted acoustic feature matrices (Python dict keyed by clip ID) |
| `split_indices.p` | Pickle | 5-fold cross-validation indices (552 train / 138 test per fold) |

**Show distribution:**
| Show | Count |
|---|---|
| FRIENDS | 356 |
| BBT (Big Bang Theory) | 280 |
| GOLDEN GIRLS | 40 |
| SARCASMOHOLICS | 14 |

**Audio features shape distribution:**
| Shape | Count | Notes |
|---|---|---|
| 283 × 11 | 155 | 11 acoustic channels |
| 283 × 12 | 508 | 12 acoustic channels (most common) |
| 283 × 13 | 22 | 13 acoustic channels |
| 283 × 14–18 | 5 | 14–18 acoustic channels (rare) |

> [!IMPORTANT]
> The `audio_features.p` file contains **pre-extracted frame-level acoustic feature matrices**, NOT raw audio waveforms. Each entry is a NumPy array of shape (283, d) where d varies from 11 to 18. These represent pre-computed acoustic features (likely openSMILE eGeMAPS or similar) over 283 time frames. **No `.wav`, `.mp3`, or `.mp4` files exist in this project.**

**Key alignment:** All 690 keys in `sarcasm_data.json` exactly match all 690 keys in `audio_features.p`. Zero mismatches.

### 2.2 MUStARD++ Dataset

**Location:** `data/mustard_plus/`

**Files present (1 total):**
| File | Size | Description |
|---|---|---|
| `mustard++_text.csv` | CSV | 6,041 rows, 12 columns |

**Columns:** `SCENE`, `KEY`, `SENTENCE`, `END_TIME`, `SPEAKER`, `SHOW`, `Sarcasm`, `Sarcasm_Type`, `Implicit_Emotion`, `Explicit_Emotion`, `Valence`, `Arousal`

**Labeling statistics:**
- Rows with sarcasm labels: **1,202** (601 sarcastic, 601 non-sarcastic)
- Rows without sarcasm labels: **4,839** (context utterances from surrounding dialogue)

**Show distribution (labeled rows):**
| Show | Count |
|---|---|
| BBT | 716 |
| FRIENDS | 354 |
| Silicon Valley (SV) | 78 |
| GOLDEN GIRLS | 40 |
| SARCASMOHOLICS | 14 |

> [!CAUTION]
> MUStARD++ contains **TEXT ONLY** in this project. No audio files, no audio features, no video files. The additional emotion annotations (`Implicit_Emotion`, `Explicit_Emotion`, `Valence`, `Arousal`) are textual/survey-based annotations, NOT extracted from audio.

---

## 3. What Is Currently Integrated and Used

### During Training (`scripts/run_experiments.py`):
- **MUStARD (690 samples):** Used for all 5-fold CV training with text embeddings (RoBERTa 768-d + sentiment probs) + audio features (55-d pooled from MUStARD feature matrices) + mismatch features (11-d).
- **MUStARD++:** Only referenced for basic statistics. NOT used for model training.

### During Live Inference (`app.py` + `sarcastone/pipeline.py`):
- The pipeline uses **rule-based sarcasm scoring** (hand-crafted weighted formula), NOT the trained sklearn classifier.
- Audio input comes from either hardcoded **AUDIO_PRESETS** (dropdown selections) or librosa-based feature extraction from uploaded files.
- The trained `sarcastone_fusion_model.joblib` model is loaded but **never used for predictions** in the web app. The `analyze()` method in `pipeline.py` computes a hand-tuned `sarcasm_score` using weighted mismatch features.

### Modality Assessment:
| Claim | Reality |
|---|---|
| "Multimodal (Text + Audio)" | ✅ Training uses both text and pre-extracted audio features from MUStARD |
| "wav2vec 2.0 embeddings" | ❌ Not used. Emotion estimation uses heuristic acoustic-to-affect mapping |
| "openSMILE eGeMAPS" | ⚠️ MUStARD's pre-extracted features were likely computed with openSMILE by the original authors, but this project does NOT run openSMILE |
| "Live multimodal inference" | ❌ Uses rule-based scoring, not trained model |
| "MUStARD++ multimodal" | ❌ MUStARD++ is text-only in this project |

---

## 4. How to Obtain Raw Audio/Video

### MUStARD Raw Video Clips
The original MUStARD dataset (Castro et al., 2019) provides raw utterance video clips via Google Drive:
- **Source:** [github.com/soujanyaporia/MUStARD](https://github.com/soujanyaporia/MUStARD)
- The README links to a Google Drive folder containing ~2GB of utterance-final video clips from TV shows.
- **Access:** Public download (may require manual Google Drive download for large files).

### MUStARD++ Raw Audio/Video
MUStARD++ (Ray et al., 2022) uses the same TV show clips as MUStARD plus additional Silicon Valley clips:
- **Source:** [github.com/cfiltnlp/MUStARD_Plus_Plus](https://github.com/cfiltnlp/MUStARD_Plus_Plus)
- Raw clips require the original video sources.

> [!NOTE]
> For this project, raw audio is NOT strictly required because MUStARD provides pre-extracted acoustic feature matrices that can be used for training. The librosa-based feature extraction in `audio_modality.py` handles real-time audio input from users.

---

## 5. What Needs to Be Fixed

1. **Replace rule-based inference with trained model predictions** – The `pipeline.py` `analyze()` method must use the trained sklearn classifier (or a new properly-trained model) instead of the hand-crafted weighted formula.
2. **Train separate text, audio, and fusion models** – Currently only the fusion model exists as a joblib file. Need separate models for text-only and audio-only predictions.
3. **Add three-mode detection** – Text-only, Audio-only, and Multimodal detection modes in the web UI.
4. **Document model evaluation honestly** – Report real train/validation/test metrics from trained models.
5. **MUStARD++ integration** – Currently text-only; can be used for text-only model training to increase training data.

---

## 6. Conclusion

The SarcasTone project has a solid foundation with properly aligned MUStARD text + pre-extracted audio features (690 samples). The core issue is that **the live inference pipeline uses rule-based scoring instead of trained model predictions**. The fix involves: (a) training dedicated text, audio, and multimodal classifiers, (b) saving them, and (c) using them for real-time predictions in the web app.
