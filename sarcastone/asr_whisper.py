"""
SarcasTone: Whisper & ASR Transcription Robustness Evaluation Module
Evaluates the impact of automatic speech recognition (ASR) errors on sarcasm detection.
"""

import random
import re

COMMON_ASR_SUBSTITUTIONS = {
    "wonderful": "wonder full",
    "privilege": "priviledge",
    "genius": "genus",
    "brilliant": "billion",
    "another": "a nother",
    "you're": "your",
    "it's": "its",
    "don't": "dont",
    "can't": "cant",
    "really": "real",
    "great": "rate",
    "perfect": "prefer",
    "fantastic": "fantistic",
    "thanks": "thank",
    "thank you": "than you"
}

def simulate_whisper_transcription(text, word_error_rate=0.12, random_seed=42):
    """
    Simulates real-world Whisper ASR transcription noise:
    - Acoustic/phonetic confusions
    - Contraction expansions / loss of apostrophes
    - Dropped unstressed words
    - Case flattening
    """
    rng = random.Random(random_seed + hash(text) % 10000)
    words = text.split()
    perturbed = []
    
    for w in words:
        lower_w = re.sub(r'[^\w\s]', '', w.lower())
        # Check phonetic substitution
        if lower_w in COMMON_ASR_SUBSTITUTIONS and rng.random() < 0.40:
            sub = COMMON_ASR_SUBSTITUTIONS[lower_w]
            perturbed.append(sub)
        elif rng.random() < (word_error_rate * 0.5):
            # Dropped unstressed particle
            if lower_w in ["a", "the", "to", "of", "and", "in", "it"]:
                continue
            else:
                perturbed.append(w)
        elif rng.random() < (word_error_rate * 0.5):
            # Minor character typo / phonetic drift
            if len(w) > 4:
                idx = rng.randint(1, len(w) - 2)
                char_swap = list(w)
                char_swap[idx] = rng.choice(["e", "a", "i", "o", "u"])
                perturbed.append("".join(char_swap))
            else:
                perturbed.append(w)
        else:
            perturbed.append(w)
            
    res = " ".join(perturbed)
    # Whisper punctuation variance
    if rng.random() < 0.25 and res.endswith("!"):
        res = res[:-1] + "."
    elif rng.random() < 0.20 and res.endswith("?"):
        res = res[:-1] + "."
    return res
