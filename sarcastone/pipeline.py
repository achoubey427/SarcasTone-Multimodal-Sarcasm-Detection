"""
SarcasTone: Updated End-to-End Multimodal Inference Pipeline
Uses TRAINED models (text, audio, multimodal) for predictions instead of rule-based scoring.
"""

import numpy as np
import os
import joblib
from sarcastone.text_modality import TextFeatureExtractor
from sarcastone.audio_modality import AudioProsodyExtractor, AUDIO_PRESETS
from sarcastone.mismatch import extract_mismatch_vector

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "sarcastone_text_model.joblib")
AUDIO_MODEL_PATH = os.path.join(MODELS_DIR, "sarcastone_audio_model.joblib")
MULTIMODAL_MODEL_PATH = os.path.join(MODELS_DIR, "sarcastone_multimodal_model.joblib")

class SarcasTonePipeline:
    def __init__(self):
        self.text_extractor = TextFeatureExtractor()
        self.audio_extractor = AudioProsodyExtractor()
        self.text_model = None
        self.audio_model = None
        self.multimodal_model = None
        self._load_models()

    def _load_models(self):
        """Load all three trained models."""
        for name, path, attr in [
            ("Text", TEXT_MODEL_PATH, "text_model"),
            ("Audio", AUDIO_MODEL_PATH, "audio_model"),
            ("Multimodal", MULTIMODAL_MODEL_PATH, "multimodal_model"),
        ]:
            if os.path.exists(path):
                try:
                    model = joblib.load(path)
                    setattr(self, attr, model)
                    print(f"✓ Loaded {name} model from {path}")
                except Exception as e:
                    print(f"✗ Failed to load {name} model: {e}")
            else:
                print(f"✗ {name} model not found at {path}")

    def _extract_text_features(self, utterance, context=None):
        """Extract text feature vector (771-d: 768 RoBERTa CLS + 3 sentiment probs)."""
        text_res = self.text_extractor.extract_features(utterance, context=context)
        sentiment_probs = text_res["sentiment_probs"]
        text_emb = text_res["embedding"]
        X_text = np.concatenate([text_emb, sentiment_probs]).reshape(1, -1)
        return X_text, sentiment_probs

    def _extract_audio_features_from_raw(self, audio_data, sample_rate=16000):
        """Extract audio feature vector from raw audio (59-d: 14 prosody + padding to 55 + 4 emotion)."""
        raw_prosody = self.audio_extractor.extract_from_raw_audio(audio_data, sample_rate)
        emotion_probs = self.audio_extractor.estimate_emotion_from_prosody(raw_prosody, is_mustard=False)
        # Pad raw prosody (14-d) to match MUStARD training format (55-d) by zero-filling
        prosody_padded = np.zeros(55, dtype=np.float32)
        prosody_padded[:len(raw_prosody)] = raw_prosody
        X_audio = np.concatenate([prosody_padded, emotion_probs]).reshape(1, -1)
        return X_audio, emotion_probs, raw_prosody

    def _extract_audio_features_from_preset(self, preset_name):
        """Extract audio feature vector from a preset."""
        preset = AUDIO_PRESETS[preset_name]
        emotion_probs = np.array(preset["emotion_probs"], dtype=np.float32)
        prosody_padded = np.zeros(55, dtype=np.float32)
        # Fill some key positions from preset summary
        ps = preset.get("prosody_summary", {})
        prosody_padded[0] = ps.get("f0_mean", 150.0)
        prosody_padded[1] = ps.get("f0_std", 20.0)
        prosody_padded[4] = ps.get("rms_mean", 0.15)
        X_audio = np.concatenate([prosody_padded, emotion_probs]).reshape(1, -1)
        return X_audio, emotion_probs

    def predict_text(self, utterance, context=None):
        """Text-only sarcasm prediction using trained text model."""
        X_text, sentiment_probs = self._extract_text_features(utterance, context)
        
        if self.text_model is None:
            return {"error": "Text model not loaded. Please train models first."}
        
        pred = int(self.text_model.predict(X_text)[0])
        try:
            proba = float(self.text_model.predict_proba(X_text)[0][1])
        except:
            proba = float(pred)
        
        neg_p, neu_p, pos_p = float(sentiment_probs[0]), float(sentiment_probs[1]), float(sentiment_probs[2])
        if pos_p > max(neg_p, neu_p):
            sent_label = "Positive"
        elif neg_p > max(pos_p, neu_p):
            sent_label = "Negative"
        else:
            sent_label = "Neutral"
        
        return {
            "prediction": "Sarcastic" if pred == 1 else "Non-Sarcastic",
            "is_sarcastic": bool(pred == 1),
            "confidence": float(abs(proba - 0.5) * 2.0 * 100.0),
            "sarcasm_probability": proba,
            "sentiment_label": sent_label,
            "sentiment_probs": {"negative": neg_p, "neutral": neu_p, "positive": pos_p},
            "model_used": "Text-Only (RoBERTa + Sentiment)"
        }

    def predict_audio(self, audio_data=None, sample_rate=16000, preset_name=None):
        """Audio-only sarcasm prediction using trained audio model."""
        if self.audio_model is None:
            return {"error": "Audio model not loaded. Please train models first."}
        
        if audio_data is not None:
            X_audio, emotion_probs, raw_prosody = self._extract_audio_features_from_raw(audio_data, sample_rate)
            prosody_summary = {
                "f0_mean": float(raw_prosody[0]),
                "f0_std": float(raw_prosody[1]),
                "rms_mean": float(raw_prosody[4]),
                "tempo": float(raw_prosody[12]) if len(raw_prosody) > 12 else 120.0
            }
            source = "uploaded_audio"
        elif preset_name and preset_name in AUDIO_PRESETS:
            X_audio, emotion_probs = self._extract_audio_features_from_preset(preset_name)
            prosody_summary = AUDIO_PRESETS[preset_name].get("prosody_summary", {})
            source = f"preset:{preset_name}"
        else:
            return {"error": "No audio input provided. Upload a file or select a preset."}
        
        pred = int(self.audio_model.predict(X_audio)[0])
        try:
            proba = float(self.audio_model.predict_proba(X_audio)[0][1])
        except:
            proba = float(pred)
        
        emo_labels = ["Neutral", "Happy", "Angry/Annoyed", "Sad/Frustrated"]
        dominant_emo = emo_labels[int(np.argmax(emotion_probs))]
        
        return {
            "prediction": "Sarcastic" if pred == 1 else "Non-Sarcastic",
            "is_sarcastic": bool(pred == 1),
            "confidence": float(abs(proba - 0.5) * 2.0 * 100.0),
            "sarcasm_probability": proba,
            "dominant_emotion": dominant_emo,
            "emotion_probs": {
                "neutral": float(emotion_probs[0]),
                "happy": float(emotion_probs[1]),
                "angry_annoyed": float(emotion_probs[2]),
                "sad_frustrated": float(emotion_probs[3])
            },
            "prosody_summary": prosody_summary,
            "audio_source": source,
            "model_used": "Audio-Only (Prosody + Emotion)"
        }

    def predict_multimodal(self, utterance, context=None, audio_data=None, sample_rate=16000, preset_name=None):
        """
        Multimodal (text + audio + mismatch) sarcasm prediction.
        Returns individual text/audio predictions AND fused prediction.
        """
        # 1. Get individual predictions
        text_result = self.predict_text(utterance, context)
        audio_result = self.predict_audio(audio_data, sample_rate, preset_name)
        
        if "error" in text_result or "error" in audio_result:
            return {
                "text_result": text_result,
                "audio_result": audio_result,
                "fused_result": {"error": "Cannot compute fusion: one or more models failed."}
            }
        
        # 2. Extract features for fusion
        X_text, sentiment_probs = self._extract_text_features(utterance, context)
        
        if audio_data is not None:
            X_audio, emotion_probs, _ = self._extract_audio_features_from_raw(audio_data, sample_rate)
        elif preset_name and preset_name in AUDIO_PRESETS:
            X_audio, emotion_probs = self._extract_audio_features_from_preset(preset_name)
        else:
            return {
                "text_result": text_result,
                "audio_result": {"error": "No audio provided"},
                "fused_result": {"error": "Cannot compute fusion without audio."}
            }
        
        # 3. Compute mismatch features
        mismatch_res = extract_mismatch_vector(sentiment_probs, emotion_probs)
        mismatch_vec = mismatch_res["feature_vector"].reshape(1, -1)
        
        # 4. Concatenate for multimodal
        X_multimodal = np.hstack([X_text, X_audio, mismatch_vec])
        
        if self.multimodal_model is not None:
            pred = int(self.multimodal_model.predict(X_multimodal)[0])
            try:
                proba = float(self.multimodal_model.predict_proba(X_multimodal)[0][1])
            except:
                proba = float(pred)
        else:
            # Fallback: average text and audio probabilities
            proba = (text_result["sarcasm_probability"] + audio_result["sarcasm_probability"]) / 2.0
            pred = 1 if proba >= 0.5 else 0
        
        fused_result = {
            "prediction": "Sarcastic" if pred == 1 else "Non-Sarcastic",
            "is_sarcastic": bool(pred == 1),
            "confidence": float(abs(proba - 0.5) * 2.0 * 100.0),
            "sarcasm_probability": proba,
            "discordance": {
                "discord_score": float(mismatch_res["abs_discordance"]),
                "agreement_score": float(mismatch_res["agreement_score"]),
                "is_discordant": bool(mismatch_res["is_discordant"])
            },
            "model_used": "Multimodal Fusion (Text + Audio + Mismatch)"
        }
        
        return {
            "text_result": text_result,
            "audio_result": audio_result,
            "fused_result": fused_result
        }

    # Backward-compatible analyze() method
    def analyze(self, utterance, context=None, audio_data=None, sample_rate=16000, audio_preset_name=None):
        """
        Backward-compatible analysis method.
        Now uses trained models instead of rule-based scoring.
        """
        mm_result = self.predict_multimodal(
            utterance, context, audio_data, sample_rate, audio_preset_name
        )
        
        text_res = mm_result.get("text_result", {})
        audio_res = mm_result.get("audio_result", {})
        fused_res = mm_result.get("fused_result", {})
        
        # Use fused result if available, else text result
        if "error" not in fused_res:
            is_sarc = fused_res["is_sarcastic"]
            prob = fused_res["sarcasm_probability"]
            conf = fused_res["confidence"]
        elif "error" not in text_res:
            is_sarc = text_res["is_sarcastic"]
            prob = text_res["sarcasm_probability"]
            conf = text_res["confidence"]
        else:
            is_sarc = False
            prob = 0.5
            conf = 0.0
        
        # Build sentiment data
        sent_probs = text_res.get("sentiment_probs", {"negative": 0.33, "neutral": 0.34, "positive": 0.33})
        sent_label = text_res.get("sentiment_label", "Neutral")
        text_polarity = sent_probs.get("positive", 0.33) - sent_probs.get("negative", 0.33)
        
        # Build emotion data
        emo_probs = audio_res.get("emotion_probs", {"neutral": 0.75, "happy": 0.1, "angry_annoyed": 0.08, "sad_frustrated": 0.07})
        dominant_emo = audio_res.get("dominant_emotion", "Neutral")
        
        # Mismatch data
        disc = fused_res.get("discordance", {})
        discord_score = disc.get("discord_score", 0.0)
        agreement_score = disc.get("agreement_score", 0.5)
        
        # Speech valence/arousal
        from sarcastone.mismatch import compute_speech_valence_arousal
        ep = np.array([emo_probs.get("neutral", 0.75), emo_probs.get("happy", 0.1),
                        emo_probs.get("angry_annoyed", 0.08), emo_probs.get("sad_frustrated", 0.07)])
        speech_valence, speech_arousal = compute_speech_valence_arousal(ep)
        
        # Explanation
        explanation = self._generate_explanation(
            utterance, sent_label, sent_probs.get("positive", 0.33),
            dominant_emo, discord_score, is_sarc,
            max(0, text_polarity) * (emo_probs.get("angry_annoyed",0) + emo_probs.get("sad_frustrated",0)),
            max(0, -text_polarity) * emo_probs.get("happy",0)
        )
        
        return {
            "is_sarcastic": is_sarc,
            "sarcasm_probability": prob,
            "confidence_percent": conf,
            "verdict": "SARCASTIC" if is_sarc else "NON-SARCASTIC (LITERAL)",
            "text_analysis": {
                "sentiment_label": sent_label,
                "sentiment_polarity": text_polarity,
                "probabilities": sent_probs
            },
            "audio_analysis": {
                "dominant_emotion": dominant_emo,
                "speech_valence": speech_valence,
                "speech_arousal": speech_arousal,
                "probabilities": emo_probs,
                "prosody_summary": audio_res.get("prosody_summary", {}),
                "description": "Processed via trained model"
            },
            "discordance_analysis": {
                "discord_score": discord_score,
                "positive_text_negative_speech_discord": max(0, text_polarity) * (emo_probs.get("angry_annoyed",0) + emo_probs.get("sad_frustrated",0)),
                "negative_text_positive_speech_discord": max(0, -text_polarity) * emo_probs.get("happy",0),
                "agreement_score": agreement_score,
                "is_discordant": bool(discord_score > 0.40)
            },
            "explanation": explanation,
            # Include individual model results
            "text_model_prediction": text_res.get("prediction", "N/A"),
            "audio_model_prediction": audio_res.get("prediction", "N/A"),
            "fusion_model_prediction": fused_res.get("prediction", "N/A")
        }

    def _generate_explanation(self, utterance, sentiment, pos_p, emotion, discord, is_sarcastic, pos_neg, neg_pos):
        if is_sarcastic:
            if pos_neg > 0.2:
                return (
                    f"Sarcasm detected: The utterance has {sentiment} wording "
                    f"(positive score: {pos_p:.2f}), but the speech prosody indicates "
                    f"{emotion} tone. This text-speech mismatch is a hallmark of sarcasm."
                )
            elif neg_pos > 0.2:
                return (
                    f"Sarcasm detected via mock-cheerful intonation: "
                    f"The text has negative semantics but the voice sounds upbeat, "
                    f"indicating ironic delivery."
                )
            else:
                return (
                    f"Sarcasm detected through combined textual and acoustic cues "
                    f"identified by the trained multimodal fusion model."
                )
        else:
            if discord < 0.35:
                return (
                    f"Non-sarcastic: The textual sentiment ({sentiment}) is concordant "
                    f"with the speaker's vocal tone ({emotion}). Both channels convey aligned affect."
                )
            else:
                return (
                    f"Non-sarcastic: Despite some acoustic variation, the trained model "
                    f"does not find sufficient evidence of sarcastic intent."
                )
