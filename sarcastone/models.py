"""
SarcasTone: Experimental Models & Multimodal Fusion Classifiers
Implements:
1. TF-IDF + Logistic Regression (Text Baseline)
2. Text Only (RoBERTa + Sentiment)
3. Audio Only (eGeMAPS Prosody + Speech Emotion)
4. Multimodal Fusion (Text + Audio)
5. Multimodal Fusion + Mismatch Features (Full SarcasTone)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def build_classifier(model_type="logistic", random_state=42):
    if model_type == "logistic":
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state)
    elif model_type == "gb":
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=random_state)
    elif model_type == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state, early_stopping=True)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])

class SarcasToneExperimentSuite:
    def __init__(self, dataset):
        self.dataset = dataset
        self.samples = dataset["samples"]
        self.splits = dataset["splits"]
        
    def prepare_feature_matrices(self, use_context=False):
        N = len(self.samples)
        y = np.array([s["label"] for s in self.samples], dtype=int)
        
        # 1. Text contextual embeddings (reduced to top PCA or raw) + sentiment probs
        if use_context:
            text_emb = np.array([s["context_emb"] for s in self.samples], dtype=np.float32)
        else:
            text_emb = np.array([s["text_emb"] for s in self.samples], dtype=np.float32)
            
        sentiment_probs = np.array([s["sentiment_probs"] for s in self.samples], dtype=np.float32)
        X_text = np.hstack([text_emb, sentiment_probs])
        
        # 2. Audio prosody + speech emotion
        prosody = np.array([s["prosody_vec"] for s in self.samples], dtype=np.float32)
        emotion = np.array([s["emotion_probs"] for s in self.samples], dtype=np.float32)
        X_audio = np.hstack([prosody, emotion])
        
        # 3. Mismatch features
        mismatch = np.array([s["mismatch_vec"] for s in self.samples], dtype=np.float32)
        
        # 4. Multimodal (Text + Audio)
        X_multimodal_base = np.hstack([X_text, X_audio])
        
        # 5. Full Multimodal + Mismatch
        X_multimodal_mismatch = np.hstack([X_text, X_audio, mismatch])
        
        return {
            "y": y,
            "text": X_text,
            "audio": X_audio,
            "mismatch": mismatch,
            "multimodal_base": X_multimodal_base,
            "multimodal_mismatch": X_multimodal_mismatch
        }

    def evaluate_5fold_cv(self, X, y, model_type="rf"):
        accs, precs, recs, f1s = [], [], [], []
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.splits):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            clf = build_classifier(model_type=model_type, random_state=42 + fold_idx)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            accs.append(accuracy_score(y_test, y_pred))
            precs.append(precision_score(y_test, y_pred, zero_division=0))
            recs.append(recall_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
            
        return {
            "accuracy": float(np.mean(accs)),
            "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)),
            "f1": float(np.mean(f1s)),
            "accuracy_std": float(np.std(accs)),
            "f1_std": float(np.std(f1s))
        }
