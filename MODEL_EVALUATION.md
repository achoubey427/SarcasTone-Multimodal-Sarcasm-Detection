# SarcasTone – Model Evaluation Report

> **Generated:** 2026-11-27
> **Dataset:** MUStARD (690 samples, 345 sarcastic / 345 non-sarcastic)
> **Evaluation:** 5-fold cross-validation (552 train / 138 test per fold)
> **Models trained on real data only – no hardcoded results**

---

## 1. Performance Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Text-Only (RoBERTa + Sentiment) | 62.17% ± 4.64% | 62.13% ± 5.67% | 63.49% ± 4.59% | 62.67% ± 4.34% |
| Audio-Only (Prosody + Emotion) | 67.68% ± 2.13% | 66.97% ± 2.62% | 69.87% ± 4.49% | 68.30% ± 2.75% |
| Multimodal Fusion (Text + Audio + Mismatch) | 66.81% ± 2.40% | 65.33% ± 3.36% | 72.19% ± 6.07% | 68.40% ± 3.18% |
| Text + Audio (No Mismatch) | 68.70% ± 3.22% | 68.55% ± 4.26% | 69.89% ± 3.78% | 69.07% ± 2.62% |

---

## Text-Only (RoBERTa + Sentiment)

- **Model type:** mlp
- **Feature dimensions:** 771
- **Best classifier:** mlp

### Confusion Matrix (aggregated over 5 folds)

| | Predicted Non-Sarcastic | Predicted Sarcastic |
|---|---|---|
| **Actual Non-Sarcastic** | 210 (TN) | 135 (FP) |
| **Actual Sarcastic** | 126 (FN) | 219 (TP) |

### Per-Fold Results

| Fold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0 | 57.25% | 54.43% | 65.15% | 59.31% |
| 1 | 65.94% | 65.33% | 70.00% | 67.59% |
| 2 | 61.59% | 63.24% | 60.56% | 61.87% |
| 3 | 57.25% | 57.35% | 56.52% | 56.93% |
| 4 | 68.84% | 70.31% | 65.22% | 67.67% |

---

## Audio-Only (Prosody + Emotion)

- **Model type:** gb
- **Feature dimensions:** 59
- **Best classifier:** gb

### Confusion Matrix (aggregated over 5 folds)

| | Predicted Non-Sarcastic | Predicted Sarcastic |
|---|---|---|
| **Actual Non-Sarcastic** | 226 (TN) | 119 (FP) |
| **Actual Sarcastic** | 104 (FN) | 241 (TP) |

### Per-Fold Results

| Fold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0 | 67.39% | 63.64% | 74.24% | 68.53% |
| 1 | 65.94% | 65.75% | 68.57% | 67.13% |
| 2 | 71.74% | 71.62% | 74.65% | 73.10% |
| 3 | 65.94% | 67.19% | 62.32% | 64.66% |
| 4 | 67.39% | 66.67% | 69.57% | 68.09% |

---

## Multimodal Fusion (Text + Audio + Mismatch)

- **Model type:** mlp
- **Feature dimensions:** 841
- **Best classifier:** mlp

### Confusion Matrix (aggregated over 5 folds)

| | Predicted Non-Sarcastic | Predicted Sarcastic |
|---|---|---|
| **Actual Non-Sarcastic** | 212 (TN) | 133 (FP) |
| **Actual Sarcastic** | 96 (FN) | 249 (TP) |

### Per-Fold Results

| Fold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0 | 64.49% | 60.24% | 75.76% | 67.11% |
| 1 | 70.29% | 70.42% | 71.43% | 70.92% |
| 2 | 68.12% | 67.09% | 74.65% | 70.67% |
| 3 | 63.77% | 64.62% | 60.87% | 62.69% |
| 4 | 67.39% | 64.29% | 78.26% | 70.59% |

---

## Text + Audio (No Mismatch)

- **Model type:** rf
- **Feature dimensions:** 830
- **Best classifier:** rf

### Confusion Matrix (aggregated over 5 folds)

| | Predicted Non-Sarcastic | Predicted Sarcastic |
|---|---|---|
| **Actual Non-Sarcastic** | 233 (TN) | 112 (FP) |
| **Actual Sarcastic** | 104 (FN) | 241 (TP) |

### Per-Fold Results

| Fold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0 | 67.39% | 63.64% | 74.24% | 68.53% |
| 1 | 73.19% | 73.24% | 74.29% | 73.76% |
| 2 | 63.77% | 63.64% | 69.01% | 66.22% |
| 3 | 68.12% | 69.23% | 65.22% | 67.16% |
| 4 | 71.01% | 73.02% | 66.67% | 69.70% |

---

## Notes

1. All results are from **real model training** on the MUStARD dataset. No results are hardcoded.
2. Text features use RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest) 768-d CLS embeddings + 3-d sentiment probabilities.
3. Audio features use pooled statistics (mean, std, max, min, range) from MUStARD's pre-extracted 283×11 acoustic feature matrices = 55-d, plus 4-d heuristic emotion probabilities.
4. Mismatch features are 11-d vectors quantifying text-speech affective discordance.
5. Multimodal fusion concatenates text (771-d) + audio (59-d) + mismatch (11-d) = 841-d total.
6. MUStARD++ (1,202 samples) was NOT used for training because it has text only (no audio features).
7. Multiple classifier types (Logistic Regression, Random Forest, Gradient Boosting, MLP) were compared; the best per modality was selected.