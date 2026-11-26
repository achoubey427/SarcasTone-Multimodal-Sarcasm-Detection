import os
import urllib.request
import json
import pickle
import pandas as pd
import numpy as np

def setup_data():
    os.makedirs("data/mustard", exist_ok=True)
    os.makedirs("data/mustard_plus", exist_ok=True)
    
    # 1. Download MUStARD sarcasm_data.json
    mustard_json_path = "data/mustard/sarcasm_data.json"
    if not os.path.exists(mustard_json_path):
        print("Downloading MUStARD sarcasm_data.json...")
        url = "https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json"
        urllib.request.urlretrieve(url, mustard_json_path)
    
    # 2. Download MUStARD audio features
    audio_feat_path = "data/mustard/audio_features.p"
    if not os.path.exists(audio_feat_path):
        print("Downloading MUStARD audio_features.p...")
        url = "https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/audio_features.p"
        try:
            urllib.request.urlretrieve(url, audio_feat_path)
        except Exception as e:
            print("Failed downloading audio_features.p:", e)

    # 3. Download MUStARD split indices
    split_path = "data/mustard/split_indices.p"
    if not os.path.exists(split_path):
        print("Downloading MUStARD split_indices.p...")
        url = "https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/split_indices.p"
        try:
            urllib.request.urlretrieve(url, split_path)
        except Exception as e:
            print("Failed downloading split_indices.p:", e)
            
    # 4. Download MUStARD++ text csv
    plus_csv_path = "data/mustard_plus/mustard++_text.csv"
    if not os.path.exists(plus_csv_path):
        print("Downloading MUStARD++ CSV...")
        url = "https://raw.githubusercontent.com/apoorva-nunna/MUStARD_Plus_Plus/main/mustard%2B%2B_text.csv"
        urllib.request.urlretrieve(url, plus_csv_path)

    print("Checking downloaded files:")
    with open(mustard_json_path, 'r', encoding='utf-8') as f:
        m_json = json.load(f)
    print(f"MUStARD items: {len(m_json)}")

    if os.path.exists(audio_feat_path):
        with open(audio_feat_path, 'rb') as f:
            aud = pickle.load(f)
            print(f"MUStARD audio_features keys: {len(aud)}")
            k0 = list(aud.keys())[0]
            print(f"Sample audio feature shape for {k0}: {np.shape(aud[k0])}")

    df_plus = pd.read_csv(plus_csv_path)
    target_plus = df_plus[df_plus['Sarcasm'].notnull() & df_plus['Sarcasm'].isin([0, 1, 0.0, 1.0])]
    print(f"MUStARD++ target labeled items: {len(target_plus)}")

if __name__ == "__main__":
    setup_data()
