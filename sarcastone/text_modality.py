"""
SarcasTone: Text Modality & Contextual Feature Extraction Module
Extracts contextual embeddings using RoBERTa and sentiment distributions via CardiffNLP.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer

SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

class TextFeatureExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model()
        self.tfidf_vectorizer = None

    def _load_model(self):
        try:
            print("Loading RoBERTa & CardiffNLP Sentiment model...")
            # Try offline / local files first for fast loading
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME, local_files_only=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
                self.model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            print("RoBERTa model successfully loaded.")
        except Exception as e:
            print(f"Warning: Failed loading HF RoBERTa model ({e}). Using lightweight fallback.")
            self.model = None

    def extract_features(self, text, context=None):
        """
        Extracts:
        - sentiment_probs: [P(neg), P(neu), P(pos)]
        - contextual_embedding: 768-d RoBERTa CLS representation
        - context_embedding: representation with dialogue history if provided
        """
        if context and isinstance(context, list) and len(context) > 0:
            full_text = " ".join(context) + " </s> " + text
        else:
            full_text = text

        if self.model is not None:
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                logits = outputs.logits
                sentiment_probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                cls_embedding = outputs.hidden_states[-1][0, 0, :].cpu().numpy()

            # CardiffNLP returns [neg, neu, pos]
            return {
                "sentiment_probs": sentiment_probs,
                "embedding": cls_embedding,
                "text": text,
                "full_text": full_text
            }
        else:
            # Deterministic heuristic fallback if offline
            neg = 0.33
            neu = 0.34
            pos = 0.33
            lower = text.lower()
            if any(w in lower for w in ["great", "wonderful", "love", "privilege", "perfect", "good", "nice"]):
                pos = 0.82
                neu = 0.12
                neg = 0.06
            elif any(w in lower for w in ["terrible", "awful", "hate", "bad", "worst", "idiot", "annoying"]):
                neg = 0.85
                neu = 0.10
                pos = 0.05
            sentiment_probs = np.array([neg, neu, pos], dtype=np.float32)
            cls_embedding = np.zeros(768, dtype=np.float32)
            return {
                "sentiment_probs": sentiment_probs,
                "embedding": cls_embedding,
                "text": text,
                "full_text": full_text
            }

    def train_tfidf(self, texts, max_features=1000):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
        return self.tfidf_vectorizer.fit_transform(texts).toarray()

    def transform_tfidf(self, texts):
        if self.tfidf_vectorizer is None:
            raise ValueError("TFIDF vectorizer is not fitted.")
        return self.tfidf_vectorizer.transform(texts).toarray()
