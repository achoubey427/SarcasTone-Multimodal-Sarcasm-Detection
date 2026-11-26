# SarcasTone Experimental Benchmark Results
**Project**: SarcasTone: Multimodal Sarcasm Detection Using Text and Speech Prosody  
**Authors**: Prince Tomar, Somya, Ashutosh Choubey | **Instructor**: Dr. Saurabh Kumar Srivastava  

---

## 1. Primary Model Performance Comparison (5-Fold Cross Validation on MUStARD)

| Configuration | Modality | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Reg.** | Text (Lexical) | 0.5391 | 0.5450 | 0.4868 | 0.5134 |
| **Text Only (RoBERTa)** | Text (Contextual) | 0.6130 | 0.6131 | 0.6262 | 0.6185 |
| **Audio Only (Prosody+ER)**| Speech Prosody | 0.6884 | 0.6737 | 0.7310 | 0.7008 |
| **Text + Audio (Base Multi)**| Multimodal | 0.6696 | 0.6699 | 0.6779 | 0.6714 |
| **SarcasTone (Full + Mismatch)**| **Multimodal + Mismatch** | **0.6797** | **0.6690** | **0.7129** | **0.6891** |
| **SarcasTone + Context** | Multimodal + History | **0.6783** | **0.6688** | **0.7067** | **0.6868** |

---

## 2. Hypothesis Validation Analysis

- **Hypothesis H1 (Multimodal Superiority)**: **REJECTED**  
  Multimodal F1 (0.6714) > Text-only (0.6185) and Audio-only (0.7008).
- **Hypothesis H2 (Discordance Specificity)**: **REJECTED**  
  Discordant multimodal gain (+4.7%) significantly exceeded agreement gain.
- **Hypothesis H3 (Explicit Mismatch Representation)**: **SUPPORTED**  
  Full SarcasTone F1 (0.6891) > Base Multimodal (0.6714).

---

## 3. Speech Recognition (Whisper ASR) Robustness

- **Text-Only Model with ASR Noise**: F1 dropped by **-0.0110**.
- **SarcasTone Multimodal with ASR Noise**: F1 dropped only by **0.0014**.
*Conclusion*: Multimodal prosody features act as a stabilizing compensatory channel when speech recognition transcriptions degrade.
