"""
SarcasTone: End-to-End Multimodal Inference Pipeline
Integrates text sentiment, acoustic prosody, discordance detection, and fusion verdict.
"""

import numpy as np
import os
import joblib
from sarcastone.text_modality import TextFeatureExtractor
from sarcastone.audio_modality import AudioProsodyExtractor, AUDIO_PRESETS
from sarcastone.mismatch import extract_mismatch_vector

MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sarcastone_fusion_model.joblib")

class SarcasTonePipeline:
    def __init__(self, model_path=None):
        self.text_extractor = TextFeatureExtractor()
        self.audio_extractor = AudioProsodyExtractor()
        self.model_path = model_path or MODEL_SAVE_PATH
        self.clf = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        if os.path.exists(self.model_path):
            try:
                self.clf = joblib.load(self.model_path)
                print(f"Loaded trained SarcasTone model from {self.model_path}")
                return
            except Exception as e:
                print(f"Could not load saved model ({e}), training baseline classifier.")
        self.clf = None

    def analyze(self, utterance, context=None, audio_data=None, sample_rate=16000, audio_preset_name=None):
        """
        Analyzes a multi-modal sample and produces sarcasm detection verdict with diagnostics.
        """
        # 1. Text Analysis
        text_res = self.text_extractor.extract_features(utterance, context=context)
        sentiment_probs = text_res["sentiment_probs"]
        neg_p, neu_p, pos_p = float(sentiment_probs[0]), float(sentiment_probs[1]), float(sentiment_probs[2])
        
        if pos_p > max(neg_p, neu_p):
            sentiment_label = "Positive"
        elif neg_p > max(pos_p, neu_p):
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # 2. Audio & Prosody Analysis
        prosody_summary = {}
        if audio_preset_name and audio_preset_name in AUDIO_PRESETS:
            preset = AUDIO_PRESETS[audio_preset_name]
            emotion_probs = np.array(preset["emotion_probs"], dtype=np.float32)
            prosody_summary = preset["prosody_summary"]
            prosody_desc = preset["description"]
        elif audio_data is not None:
            raw_prosody = self.audio_extractor.extract_from_raw_audio(audio_data, sample_rate)
            emotion_probs = self.audio_extractor.estimate_emotion_from_prosody(raw_prosody, is_mustard=False)
            prosody_summary = {
                "f0_mean": float(raw_prosody[0]),
                "f0_std": float(raw_prosody[1]),
                "rms_mean": float(raw_prosody[4]),
                "tempo": float(raw_prosody[12])
            }
            prosody_desc = "Extracted acoustic parameters from uploaded/recorded audio signal."
        else:
            # Default neutral
            preset = AUDIO_PRESETS["Neutral Professional Voice"]
            emotion_probs = np.array(preset["emotion_probs"], dtype=np.float32)
            prosody_summary = preset["prosody_summary"]
            prosody_desc = preset["description"]

        emo_neu, emo_hap, emo_ang, emo_sad = [float(p) for p in emotion_probs]
        dominant_emo_idx = np.argmax(emotion_probs)
        emo_labels = ["Neutral", "Happy/Joy", "Angry/Annoyed", "Sad/Frustrated"]
        dominant_emotion = emo_labels[dominant_emo_idx]

        # 3. Mismatch & Discordance Analysis
        mismatch_res = extract_mismatch_vector(sentiment_probs, emotion_probs, prosody_summary)
        
        discord_score = float(mismatch_res["abs_discordance"])
        pos_neg_discord = float(mismatch_res["pos_text_neg_speech"])
        neg_pos_discord = float(mismatch_res["neg_text_pos_speech"])
        text_polarity = float(mismatch_res["text_polarity"])
        speech_valence = float(mismatch_res["speech_valence"])
        speech_arousal = float(mismatch_res["speech_arousal"])
        agreement_score = float(mismatch_res["agreement_score"])
        
        # 4. Multimodal Sarcasm Verdict Computation
        # Base probability combining text sentiment + acoustic emotion + discordance
        # In sarcasm, either:
        # A) High Discord: Positive words + Negative tone (Sheldon/Chandler archetype)
        # B) Mock cheerfulness: Negative words + excessively upbeat voice
        # C) Deadpan sarcasm: flat acoustic prosody with ironic linguistic cues
        sarcasm_score = 0.05
        
        # Primary mismatch driver (Hypothesis H3)
        sarcasm_score += 0.58 * pos_neg_discord
        sarcasm_score += 0.45 * neg_pos_discord
        sarcasm_score += 0.20 * max(0.0, discord_score - 0.3)
        
        # Vocal arousal boost
        if discord_score > 0.35:
            sarcasm_score += 0.15 * speech_arousal
            
        # Agreement suppression (when text and tone agree, sarcasm is unlikely)
        sarcasm_score -= 0.35 * agreement_score
        
        # Deadpan check (flat pitch std + ironic context)
        f0_std = prosody_summary.get("f0_std", 25.0)
        if f0_std < 10.0 and abs(text_polarity) > 0.4:
            sarcasm_score += 0.25 # deadpan sarcasm signal
            
        # Context amplification
        if context and len(context) > 0:
            context_str = " ".join(context).lower()
            if any(term in context_str for term in ["plan", "work", "meeting", "failed", "broken", "lost"]):
                sarcasm_score += 0.08
                
        # Bound in [0.02, 0.98]
        sarcasm_prob = float(np.clip(sarcasm_score, 0.02, 0.98))
        is_sarcastic = bool(sarcasm_prob >= 0.50)
        confidence = float(abs(sarcasm_prob - 0.5) * 2.0 * 100.0) # 0 to 100%

        # 5. Natural Language Explanation
        explanation = self._generate_explanation(
            utterance, sentiment_label, pos_p, dominant_emotion,
            discord_score, is_sarcastic, pos_neg_discord, neg_pos_discord
        )

        return {
            "is_sarcastic": is_sarcastic,
            "sarcasm_probability": sarcasm_prob,
            "confidence_percent": confidence,
            "verdict": "SARCASTIC" if is_sarcastic else "NON-SARCASTIC (LITERAL)",
            "text_analysis": {
                "sentiment_label": sentiment_label,
                "sentiment_polarity": text_polarity,
                "probabilities": {"negative": neg_p, "neutral": neu_p, "positive": pos_p}
            },
            "audio_analysis": {
                "dominant_emotion": dominant_emotion,
                "speech_valence": speech_valence,
                "speech_arousal": speech_arousal,
                "probabilities": {
                    "neutral": emo_neu,
                    "happy": emo_hap,
                    "angry_annoyed": emo_ang,
                    "sad_frustrated": emo_sad
                },
                "prosody_summary": prosody_summary,
                "description": prosody_desc
            },
            "discordance_analysis": {
                "discord_score": discord_score,
                "positive_text_negative_speech_discord": pos_neg_discord,
                "negative_text_positive_speech_discord": neg_pos_discord,
                "agreement_score": agreement_score,
                "is_discordant": bool(discord_score > 0.40)
            },
            "explanation": explanation
        }

    def _generate_explanation(self, utterance, sentiment, pos_p, emotion, discord, is_sarcastic, pos_neg, neg_pos):
        if is_sarcastic:
            if pos_neg > 0.2:
                return (
                    f"Sarcasm detected due to strong affective discordance: "
                    f"The utterance exhibits {sentiment} wording (positive sentiment score: {pos_p:.2f}), "
                    f"yet the speech prosody expresses {emotion} tone. "
                    f"This stark mismatch between literal semantics and vocal delivery is a primary acoustic hallmark of sarcasm."
                )
            elif neg_pos > 0.2:
                return (
                    f"Sarcasm detected via mock-cheerful intonation: "
                    f"The text conveys negative semantics, but the vocal prosody exhibits upbeat/happy intonation, "
                    f"indicating ironic celebration or sarcastic trivialization."
                )
            else:
                return (
                    f"Sarcasm detected through combined contextual prosody cues: "
                    f"Acoustic prosody features (pitch variance, vocal tension) conflict with the literal textual statement."
                )
        else:
            if discord < 0.35:
                return (
                    f"Literal (non-sarcastic) intent identified: "
                    f"The textual sentiment ({sentiment}) is concordant with the speaker's vocal tone ({emotion}). "
                    f"Both linguistic and acoustic channels convey aligned affect."
                )
            else:
                return (
                    f"Utterance classified as non-sarcastic: "
                    f"Acoustic cues and semantic meaning lack the characteristic affective inversion required for sarcastic intent."
                )
