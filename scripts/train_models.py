"""
SarcasTone: Train Text, Audio, and Multimodal Fusion Models
Trains three separate classifiers on MUSTARD data and saves them.
Also generates MODEL_EVALUATION.md with real metrics.
"""

import os, sys, json, pickle, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold
import joblib

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

MODELS_DIR = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "data", "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_PATH = os.path.join(BASE, "data", "sarcastone_features_cache.pkl")

def load_dataset():
    """Load cached features or rebuild."""
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached features from {CACHE_PATH}...")
        with open(CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        print("No cache found. Building from scratch...")
        from sarcastone.text_modality import TextFeatureExtractor
        from sarcastone.dataset import DatasetManager
        text_ext = TextFeatureExtractor()
        dm = DatasetManager(text_extractor=text_ext)
        return dm.build_or_load_dataset(force_recompute=True)

def prepare_features(dataset):
    """Extract feature matrices from dataset samples."""
    samples = dataset['samples']
    splits = dataset['splits']
    keys = dataset['keys']
    
    N = len(samples)
    y = np.array([s['label'] for s in samples], dtype=int)
    
    # Text features: 768-d RoBERTa CLS + 3-d sentiment probs = 771-d
    text_emb = np.array([s['text_emb'] for s in samples], dtype=np.float32)
    sentiment_probs = np.array([s['sentiment_probs'] for s in samples], dtype=np.float32)
    X_text = np.hstack([text_emb, sentiment_probs])
    
    # Audio features: 55-d pooled prosody + 4-d emotion probs = 59-d
    prosody = np.array([s['prosody_vec'] for s in samples], dtype=np.float32)
    emotion = np.array([s['emotion_probs'] for s in samples], dtype=np.float32)
    X_audio = np.hstack([prosody, emotion])
    
    # Mismatch features: 11-d
    mismatch = np.array([s['mismatch_vec'] for s in samples], dtype=np.float32)
    
    # Multimodal: text + audio + mismatch
    X_multimodal = np.hstack([X_text, X_audio, mismatch])
    
    print(f"Feature dimensions:")
    print(f"  Text:       {X_text.shape}")
    print(f"  Audio:      {X_audio.shape}")
    print(f"  Mismatch:   {mismatch.shape}")
    print(f"  Multimodal: {X_multimodal.shape}")
    print(f"  Labels:     {y.shape} (sarcastic={y.sum()}, non-sarcastic={len(y)-y.sum()})")
    
    return X_text, X_audio, mismatch, X_multimodal, y, splits

def build_model(model_type="gb", random_state=42):
    """Build a sklearn pipeline with scaler + classifier."""
    if model_type == "logistic":
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=random_state)
    elif model_type == "gb":
        clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, min_samples_leaf=5, random_state=random_state)
    elif model_type == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800, random_state=random_state, early_stopping=True, validation_fraction=0.15)
    else:
        clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=random_state)
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])

def evaluate_5fold(X, y, splits, model_type="gb", name="Model"):
    """Run 5-fold CV and return aggregated metrics."""
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_metrics = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        clf = build_model(model_type=model_type, random_state=42+fold_idx)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
        except:
            y_proba = y_pred.astype(float)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        fold_metrics.append({
            'fold': fold_idx,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
        
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.extend(y_proba.tolist())
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    
    avg_metrics = {
        'accuracy': float(np.mean([f['accuracy'] for f in fold_metrics])),
        'accuracy_std': float(np.std([f['accuracy'] for f in fold_metrics])),
        'precision': float(np.mean([f['precision'] for f in fold_metrics])),
        'precision_std': float(np.std([f['precision'] for f in fold_metrics])),
        'recall': float(np.mean([f['recall'] for f in fold_metrics])),
        'recall_std': float(np.std([f['recall'] for f in fold_metrics])),
        'f1': float(np.mean([f['f1'] for f in fold_metrics])),
        'f1_std': float(np.std([f['f1'] for f in fold_metrics])),
        'confusion_matrix': cm.tolist(),
        'fold_metrics': fold_metrics
    }
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {avg_metrics['accuracy']*100:.2f}% ± {avg_metrics['accuracy_std']*100:.2f}%")
    print(f"  Precision: {avg_metrics['precision']*100:.2f}% ± {avg_metrics['precision_std']*100:.2f}%")
    print(f"  Recall:    {avg_metrics['recall']*100:.2f}% ± {avg_metrics['recall_std']*100:.2f}%")
    print(f"  F1 Score:  {avg_metrics['f1']*100:.2f}% ± {avg_metrics['f1_std']*100:.2f}%")
    print(f"  Confusion Matrix (aggregated):")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    
    return avg_metrics

def train_and_save_final_model(X, y, model_type, save_path, name):
    """Train on ALL data and save the final model."""
    clf = build_model(model_type=model_type, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, save_path)
    print(f"  Saved {name} → {save_path}")
    return clf

def select_best_model_type(X, y, splits, name):
    """Try multiple model types and return the best one by F1."""
    best_f1 = -1
    best_type = "gb"
    
    for mtype in ["logistic", "rf", "gb", "mlp"]:
        metrics = evaluate_5fold(X, y, splits, model_type=mtype, name=f"{name} ({mtype})")
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_type = mtype
            best_metrics = metrics
    
    print(f"\n  Best model for {name}: {best_type} (F1={best_f1*100:.2f}%)")
    return best_type, best_metrics

def generate_evaluation_report(all_results, output_path):
    """Generate MODEL_EVALUATION.md."""
    
    lines = []
    lines.append("# SarcasTone – Model Evaluation Report\n")
    lines.append(f"> **Generated:** 2026-11-27")
    lines.append(f"> **Dataset:** MUStARD (690 samples, 345 sarcastic / 345 non-sarcastic)")
    lines.append(f"> **Evaluation:** 5-fold cross-validation (552 train / 138 test per fold)")
    lines.append(f"> **Models trained on real data only – no hardcoded results**\n")
    lines.append("---\n")
    
    # Summary table
    lines.append("## 1. Performance Summary\n")
    lines.append("| Model | Accuracy | Precision | Recall | F1 Score |")
    lines.append("|---|---|---|---|---|")
    
    for name, res in all_results.items():
        m = res['metrics']
        lines.append(f"| {name} | {m['accuracy']*100:.2f}% ± {m['accuracy_std']*100:.2f}% | {m['precision']*100:.2f}% ± {m['precision_std']*100:.2f}% | {m['recall']*100:.2f}% ± {m['recall_std']*100:.2f}% | {m['f1']*100:.2f}% ± {m['f1_std']*100:.2f}% |")
    
    lines.append("")
    
    # Detailed per-model sections
    for name, res in all_results.items():
        m = res['metrics']
        cm = m['confusion_matrix']
        lines.append(f"---\n")
        lines.append(f"## {name}\n")
        lines.append(f"- **Model type:** {res['model_type']}")
        lines.append(f"- **Feature dimensions:** {res['feature_dim']}")
        lines.append(f"- **Best classifier:** {res['model_type']}\n")
        
        lines.append("### Confusion Matrix (aggregated over 5 folds)\n")
        lines.append("| | Predicted Non-Sarcastic | Predicted Sarcastic |")
        lines.append("|---|---|---|")
        lines.append(f"| **Actual Non-Sarcastic** | {cm[0][0]} (TN) | {cm[0][1]} (FP) |")
        lines.append(f"| **Actual Sarcastic** | {cm[1][0]} (FN) | {cm[1][1]} (TP) |\n")
        
        lines.append("### Per-Fold Results\n")
        lines.append("| Fold | Accuracy | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|---|")
        for fm in m['fold_metrics']:
            lines.append(f"| {fm['fold']} | {fm['accuracy']*100:.2f}% | {fm['precision']*100:.2f}% | {fm['recall']*100:.2f}% | {fm['f1']*100:.2f}% |")
        lines.append("")
    
    # Notes
    lines.append("---\n")
    lines.append("## Notes\n")
    lines.append("1. All results are from **real model training** on the MUStARD dataset. No results are hardcoded.")
    lines.append("2. Text features use RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest) 768-d CLS embeddings + 3-d sentiment probabilities.")
    lines.append("3. Audio features use pooled statistics (mean, std, max, min, range) from MUStARD's pre-extracted 283×11 acoustic feature matrices = 55-d, plus 4-d heuristic emotion probabilities.")
    lines.append("4. Mismatch features are 11-d vectors quantifying text-speech affective discordance.")
    lines.append("5. Multimodal fusion concatenates text (771-d) + audio (59-d) + mismatch (11-d) = 841-d total.")
    lines.append("6. MUStARD++ (1,202 samples) was NOT used for training because it has text only (no audio features).")
    lines.append("7. Multiple classifier types (Logistic Regression, Random Forest, Gradient Boosting, MLP) were compared; the best per modality was selected.")
    
    report = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nEvaluation report saved to {output_path}")

def main():
    print("="*70)
    print("SarcasTone Model Training Pipeline")
    print("="*70)
    
    # 1. Load data
    dataset = load_dataset()
    X_text, X_audio, X_mismatch, X_multimodal, y, splits = prepare_features(dataset)
    
    all_results = {}
    
    # 2. Text-Only Model
    print("\n" + "#"*70)
    print("# TRAINING TEXT-ONLY MODEL")
    print("#"*70)
    best_text_type, text_metrics = select_best_model_type(X_text, y, splits, "Text-Only")
    text_model_path = os.path.join(MODELS_DIR, "sarcastone_text_model.joblib")
    train_and_save_final_model(X_text, y, best_text_type, text_model_path, "Text-Only")
    all_results["Text-Only (RoBERTa + Sentiment)"] = {
        'metrics': text_metrics, 'model_type': best_text_type, 'feature_dim': X_text.shape[1]
    }
    
    # 3. Audio-Only Model
    print("\n" + "#"*70)
    print("# TRAINING AUDIO-ONLY MODEL")
    print("#"*70)
    best_audio_type, audio_metrics = select_best_model_type(X_audio, y, splits, "Audio-Only")
    audio_model_path = os.path.join(MODELS_DIR, "sarcastone_audio_model.joblib")
    train_and_save_final_model(X_audio, y, best_audio_type, audio_model_path, "Audio-Only")
    all_results["Audio-Only (Prosody + Emotion)"] = {
        'metrics': audio_metrics, 'model_type': best_audio_type, 'feature_dim': X_audio.shape[1]
    }
    
    # 4. Multimodal Fusion Model (Text + Audio + Mismatch)
    print("\n" + "#"*70)
    print("# TRAINING MULTIMODAL FUSION MODEL")
    print("#"*70)
    best_mm_type, mm_metrics = select_best_model_type(X_multimodal, y, splits, "Multimodal Fusion")
    mm_model_path = os.path.join(MODELS_DIR, "sarcastone_multimodal_model.joblib")
    train_and_save_final_model(X_multimodal, y, best_mm_type, mm_model_path, "Multimodal Fusion")
    all_results["Multimodal Fusion (Text + Audio + Mismatch)"] = {
        'metrics': mm_metrics, 'model_type': best_mm_type, 'feature_dim': X_multimodal.shape[1]
    }
    
    # 5. Also add text+audio (without mismatch) for comparison
    X_text_audio = np.hstack([X_text, X_audio])
    print("\n" + "#"*70)
    print("# TRAINING TEXT+AUDIO BASE MODEL (without mismatch)")
    print("#"*70)
    best_base_type, base_metrics = select_best_model_type(X_text_audio, y, splits, "Text+Audio Base")
    all_results["Text + Audio (No Mismatch)"] = {
        'metrics': base_metrics, 'model_type': best_base_type, 'feature_dim': X_text_audio.shape[1]
    }
    
    # 6. Generate evaluation report
    eval_report_path = os.path.join(BASE, "MODEL_EVALUATION.md")
    generate_evaluation_report(all_results, eval_report_path)
    
    # 7. Save comprehensive results JSON
    results_json = {}
    for name, res in all_results.items():
        results_json[name] = {
            'accuracy': res['metrics']['accuracy'],
            'precision': res['metrics']['precision'],
            'recall': res['metrics']['recall'],
            'f1': res['metrics']['f1'],
            'model_type': res['model_type'],
            'feature_dim': int(res['feature_dim'])
        }
    
    results_path = os.path.join(RESULTS_DIR, "trained_model_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results JSON saved to {results_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nSaved models:")
    print(f"  Text:       {text_model_path}")
    print(f"  Audio:      {audio_model_path}")
    print(f"  Multimodal: {mm_model_path}")

if __name__ == "__main__":
    main()
