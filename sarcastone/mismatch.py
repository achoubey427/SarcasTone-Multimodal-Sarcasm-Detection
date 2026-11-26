"""
SarcasTone: Mismatch & Discordance Feature Extraction Module
Quantifies the clash between textual sentiment and speech emotion/prosody.
"""

import numpy as np

def compute_text_polarity(sentiment_probs):
    """
    Computes continuous text sentiment polarity in [-1.0, 1.0].
    sentiment_probs: [P(neg), P(neu), P(pos)]
    """
    neg, neu, pos = sentiment_probs[0], sentiment_probs[1], sentiment_probs[2]
    # Continuous polarity weighted by confidence
    polarity = float(pos - neg)
    return polarity

def compute_speech_valence_arousal(emotion_probs):
    """
    Computes speech valence and arousal from IEMOCAP 4-class emotion probabilities:
    emotion_probs: [P(neu), P(hap), P(ang), P(sad)]
    Valence: positive for happy, negative for angry/sad in [-1.0, 1.0]
    Arousal: high for happy/angry, low for neutral/sad in [0.0, 1.0]
    """
    neu, hap, ang, sad = emotion_probs[0], emotion_probs[1], emotion_probs[2], emotion_probs[3]
    # Valence: Happy is positive (+1), Angry is negative (-0.8), Sad is negative (-0.9), Neutral is 0.0
    valence = float(hap - (0.8 * ang + 0.9 * sad))
    # Arousal: Angry and Happy are high arousal, Neutral and Sad are lower arousal
    arousal = float(0.9 * ang + 0.85 * hap + 0.3 * sad + 0.1 * neu)
    return valence, arousal

def extract_mismatch_vector(sentiment_probs, emotion_probs, prosody_summary=None):
    """
    Extracts explicit mismatch feature vector representing discordance between
    textual sentiment and speech acoustic affect.
    
    Returns:
        dict containing scalar metrics and a 1D numpy feature vector.
    """
    neg, neu, pos = sentiment_probs[0], sentiment_probs[1], sentiment_probs[2]
    e_neu, e_hap, e_ang, e_sad = emotion_probs[0], emotion_probs[1], emotion_probs[2], emotion_probs[3]
    
    text_polarity = compute_text_polarity(sentiment_probs)
    speech_valence, speech_arousal = compute_speech_valence_arousal(emotion_probs)
    
    # 1. Absolute Valence-Sentiment Discordance
    abs_discordance = abs(text_polarity - speech_valence)
    
    # 2. Canonical Sarcastic Discord: Positive words + Negative (Angry/Sad) voice
    # e.g., "Wonderful, another meeting!" (text positive, voice angry/irritated)
    pos_text_neg_speech = max(0.0, text_polarity) * (e_ang + e_sad)
    
    # 3. Euphemistic / Mocking Discord: Negative words + Upbeat/Happy voice
    # e.g., "Oh how terrible!" said with cheerful laughing intonation
    neg_text_pos_speech = max(0.0, -text_polarity) * e_hap
    
    # 4. Sentiment vs Affect Product (negative indicates opposing directions)
    interaction_polarity = text_polarity * speech_valence
    
    # 5. Agreement Score (high when both are aligned, low when conflicting)
    # Both positive: pos * hap; Both negative: neg * (ang + sad); Both neutral: neu * e_neu
    agreement_score = (pos * e_hap) + (neg * (e_ang + e_sad)) + (neu * e_neu)
    
    # 6. High-Arousal Discord: Discordance amplified by acoustic energy / arousal
    arousal_weighted_discord = abs_discordance * speech_arousal
    
    # 7. Neutrality / Uncertainty dampening
    neutral_factor = max(neu, e_neu)
    
    # 8. Contrast ratio
    contrast_ratio = (pos_text_neg_speech + neg_text_pos_speech) / (agreement_score + 1e-5)
    
    features = np.array([
        abs_discordance,
        pos_text_neg_speech,
        neg_text_pos_speech,
        interaction_polarity,
        agreement_score,
        arousal_weighted_discord,
        speech_arousal,
        speech_valence,
        text_polarity,
        neutral_factor,
        contrast_ratio
    ], dtype=np.float32)
    
    is_discordant = bool(abs_discordance > 0.4 and (pos_text_neg_speech > 0.15 or neg_text_pos_speech > 0.15))
    is_agreement = bool(agreement_score > 0.35 and interaction_polarity > 0.1)
    
    return {
        "text_polarity": text_polarity,
        "speech_valence": speech_valence,
        "speech_arousal": speech_arousal,
        "abs_discordance": abs_discordance,
        "pos_text_neg_speech": pos_text_neg_speech,
        "neg_text_pos_speech": neg_text_pos_speech,
        "interaction_polarity": interaction_polarity,
        "agreement_score": agreement_score,
        "is_discordant": is_discordant,
        "is_agreement": is_agreement,
        "feature_vector": features
    }
