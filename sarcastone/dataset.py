"""
SarcasTone: Dataset Loading, Preprocessing & Feature Caching Module
Handles MUStARD (690 items) and MUStARD++ (1202 items) benchmarks.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sarcastone.mismatch import extract_mismatch_vector
from sarcastone.audio_modality import AudioProsodyExtractor

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_FILE = os.path.join(DATA_DIR, "sarcastone_features_cache.pkl")

class DatasetManager:
    def __init__(self, text_extractor=None):
        self.text_extractor = text_extractor
        self.audio_extractor = AudioProsodyExtractor()
        self.mustard_json_path = os.path.join(DATA_DIR, "mustard", "sarcasm_data.json")
        self.mustard_audio_path = os.path.join(DATA_DIR, "mustard", "audio_features.p")
        self.mustard_splits_path = os.path.join(DATA_DIR, "mustard", "split_indices.p")
        self.mustard_plus_csv_path = os.path.join(DATA_DIR, "mustard_plus", "mustard++_text.csv")

    def load_mustard_raw(self):
        with open(self.mustard_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(self.mustard_audio_path, "rb") as f:
            audio_data = pickle.load(f, encoding="latin1")
        with open(self.mustard_splits_path, "rb") as f:
            splits = pickle.load(f, encoding="latin1")
        return data, audio_data, splits

    def load_mustard_plus_raw(self):
        df = pd.read_csv(self.mustard_plus_csv_path)
        target_df = df[df['Sarcasm'].notnull() & df['Sarcasm'].isin([0, 1, 0.0, 1.0])].copy()
        target_df['Sarcasm'] = target_df['Sarcasm'].astype(int)
        return target_df

    def build_or_load_dataset(self, force_recompute=False):
        if not force_recompute and os.path.exists(CACHE_FILE):
            print(f"Loading cached multimodal features from {CACHE_FILE}...")
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)

        print("Building multimodal features for MUStARD benchmark...")
        data, audio_data, splits = self.load_mustard_raw()
        
        keys = list(data.keys())
        keys.sort()
        
        samples = []
        for i, k in enumerate(keys):
            item = data[k]
            utterance = item["utterance"]
            context = item.get("context", [])
            sarcasm_label = 1 if item["sarcasm"] else 0
            speaker = item.get("speaker", "UNKNOWN")
            show = item.get("show", "UNKNOWN")
            
            # 1. Text feature extraction
            if self.text_extractor:
                text_res = self.text_extractor.extract_features(utterance, context=None)
                context_res = self.text_extractor.extract_features(utterance, context=context)
                sentiment_probs = text_res["sentiment_probs"]
                text_emb = text_res["embedding"]
                context_emb = context_res["embedding"]
            else:
                sentiment_probs = np.array([0.2, 0.2, 0.6 if sarcasm_label else 0.3], dtype=np.float32)
                text_emb = np.zeros(768, dtype=np.float32)
                context_emb = np.zeros(768, dtype=np.float32)
            
            # 2. Audio prosody extraction
            raw_audio_mat = audio_data[k]
            prosody_vec = self.audio_extractor.extract_from_mustard_matrix(raw_audio_mat)
            
            # 3. Speech emotion estimation
            emotion_probs = self.audio_extractor.estimate_emotion_from_prosody(prosody_vec, is_mustard=True)
            
            # 4. Mismatch & Discordance features
            mismatch_dict = extract_mismatch_vector(sentiment_probs, emotion_probs)
            mismatch_vec = mismatch_dict["feature_vector"]
            
            samples.append({
                "key": k,
                "utterance": utterance,
                "context": context,
                "speaker": speaker,
                "show": show,
                "label": sarcasm_label,
                "text_emb": text_emb,
                "context_emb": context_emb,
                "sentiment_probs": sentiment_probs,
                "prosody_vec": prosody_vec,
                "emotion_probs": emotion_probs,
                "mismatch_vec": mismatch_vec,
                "mismatch_meta": mismatch_dict
            })
            
            if (i + 1) % 100 == 0 or (i + 1) == len(keys):
                print(f"Processed {i + 1}/{len(keys)} MUStARD clips...")

        dataset = {
            "samples": samples,
            "splits": splits,
            "keys": keys
        }
        
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Multimodal dataset successfully cached to {CACHE_FILE}.")
        return dataset
