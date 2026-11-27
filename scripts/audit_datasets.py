"""Deep audit of MUSTARD and MUSTARD++ datasets."""
import json, pickle, os, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("SARCASTONE DATASET AUDIT")
print("="*70)

# 1. MUSTARD
print("\n" + "="*70)
print("1. MUSTARD DATASET")
print("="*70)

mdir = os.path.join(BASE, "data", "mustard")
print(f"\nDirectory: {mdir}")
print(f"Exists: {os.path.isdir(mdir)}")

if os.path.isdir(mdir):
    all_files = []
    for root, dirs, files in os.walk(mdir):
        for f in files:
            fp = os.path.join(root, f)
            all_files.append((fp, os.path.getsize(fp), os.path.splitext(f)[1].lower()))
    
    print(f"\nTotal files in data/mustard/: {len(all_files)}")
    exts = {}
    for fp, sz, ext in all_files:
        exts[ext] = exts.get(ext, 0) + 1
    print(f"File extensions: {exts}")
    
    audio_exts = {'.wav','.mp3','.flac','.ogg','.m4a','.aac'}
    video_exts = {'.mp4','.avi','.mkv','.mov','.webm','.flv'}
    audio_files = [f for f in all_files if f[2] in audio_exts]
    video_files = [f for f in all_files if f[2] in video_exts]
    print(f"\nActual audio files (.wav/.mp3/etc): {len(audio_files)}")
    print(f"Actual video files (.mp4/.avi/etc): {len(video_files)}")
    
    # sarcasm_data.json
    json_path = os.path.join(mdir, "sarcasm_data.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            md = json.load(f)
        sarc = sum(1 for v in md.values() if v.get('sarcasm'))
        print(f"\nsarcasm_data.json:")
        print(f"  Total items: {len(md)}")
        print(f"  Sarcastic: {sarc}")
        print(f"  Non-sarcastic: {len(md)-sarc}")
        print(f"  Fields per item: {list(list(md.values())[0].keys())}")
        print(f"  Sample keys (first 5): {list(md.keys())[:5]}")
        # Shows
        shows = {}
        for v in md.values():
            s = v.get('show','unknown')
            shows[s] = shows.get(s,0)+1
        print(f"  Shows: {shows}")
    
    # audio_features.p
    af_path = os.path.join(mdir, "audio_features.p")
    if os.path.exists(af_path):
        with open(af_path, 'rb') as f:
            af = pickle.load(f, encoding='latin1')
        print(f"\naudio_features.p:")
        print(f"  Type: {type(af).__name__}")
        if isinstance(af, dict):
            print(f"  Keys count: {len(af)}")
            shapes = {}
            for k,v in af.items():
                if hasattr(v,'shape'):
                    shapes[v.shape] = shapes.get(v.shape,0)+1
            print(f"  Shape distribution: {shapes}")
            # Check alignment with json
            if os.path.exists(json_path):
                json_keys = set(md.keys())
                af_keys = set(af.keys())
                print(f"  Keys in both JSON and audio_features: {len(json_keys & af_keys)}")
                print(f"  Keys in JSON only: {len(json_keys - af_keys)}")
                print(f"  Keys in audio_features only: {len(af_keys - json_keys)}")
    
    # split_indices.p
    si_path = os.path.join(mdir, "split_indices.p")
    if os.path.exists(si_path):
        with open(si_path, 'rb') as f:
            si = pickle.load(f, encoding='latin1')
        print(f"\nsplit_indices.p:")
        print(f"  Type: {type(si).__name__}, Length: {len(si)}")
        for i,fold in enumerate(si):
            if isinstance(fold, (list,tuple)) and len(fold)==2:
                print(f"  Fold {i}: train={len(fold[0])}, test={len(fold[1])}")

# 2. MUSTARD++
print("\n" + "="*70)
print("2. MUSTARD++ DATASET")
print("="*70)

mpdir = os.path.join(BASE, "data", "mustard_plus")
print(f"\nDirectory: {mpdir}")
print(f"Exists: {os.path.isdir(mpdir)}")

if os.path.isdir(mpdir):
    all_files_pp = []
    for root, dirs, files in os.walk(mpdir):
        for f in files:
            fp = os.path.join(root, f)
            all_files_pp.append((fp, os.path.getsize(fp), os.path.splitext(f)[1].lower()))
    
    print(f"\nTotal files in data/mustard_plus/: {len(all_files_pp)}")
    exts_pp = {}
    for fp, sz, ext in all_files_pp:
        exts_pp[ext] = exts_pp.get(ext, 0) + 1
    print(f"File extensions: {exts_pp}")
    
    audio_pp = [f for f in all_files_pp if f[2] in audio_exts]
    video_pp = [f for f in all_files_pp if f[2] in video_exts]
    print(f"\nActual audio files: {len(audio_pp)}")
    print(f"Actual video files: {len(video_pp)}")
    
    csv_path = os.path.join(mpdir, "mustard++_text.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\nmustard++_text.csv:")
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        labeled = df[df['Sarcasm'].notna()]
        print(f"  Rows with Sarcasm label: {len(labeled)}")
        if len(labeled) > 0:
            vc = labeled['Sarcasm'].value_counts()
            print(f"  Sarcasm distribution: {dict(vc)}")
            # Check unique shows
            if 'SHOW' in df.columns:
                shows_pp = dict(labeled['SHOW'].value_counts())
                print(f"  Shows (labeled): {shows_pp}")
        unlabeled = df[df['Sarcasm'].isna()]
        print(f"  Rows WITHOUT Sarcasm label: {len(unlabeled)}")

# 3. Models directory
print("\n" + "="*70)
print("3. EXISTING MODELS")
print("="*70)

models_dir = os.path.join(BASE, "models")
if os.path.isdir(models_dir):
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            fp = os.path.join(root, f)
            print(f"  {f} ({os.path.getsize(fp)} bytes)")
else:
    print("  No models/ directory found")

# 4. Results
print("\n" + "="*70)
print("4. BENCHMARK RESULTS")
print("="*70)
results_path = os.path.join(BASE, "data", "results", "benchmark_results.json")
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
    for k,v in results.items():
        if isinstance(v, dict) and 'accuracy' in v:
            print(f"  {k}: Acc={v['accuracy']*100:.2f}%, F1={v.get('f1',0)*100:.2f}%")
        else:
            print(f"  {k}: {type(v).__name__}")
else:
    print("  No benchmark_results.json found")

# 5. SarcasTone source modules
print("\n" + "="*70)
print("5. SARCASTONE SOURCE MODULES")
print("="*70)
st_dir = os.path.join(BASE, "sarcastone")
if os.path.isdir(st_dir):
    for f in os.listdir(st_dir):
        fp = os.path.join(st_dir, f)
        if os.path.isfile(fp):
            print(f"  {f} ({os.path.getsize(fp)} bytes)")
else:
    print("  No sarcastone/ directory found")

# 6. Feature cache
print("\n" + "="*70)
print("6. FEATURE CACHE")
print("="*70)
cache_path = os.path.join(BASE, "data", "sarcastone_features_cache.pkl")
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    print(f"  Type: {type(cache).__name__}")
    if isinstance(cache, dict):
        print(f"  Keys: {list(cache.keys())}")
        for k,v in cache.items():
            if hasattr(v, 'shape'):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, (list, np.ndarray)):
                print(f"    {k}: len={len(v)}")
else:
    print("  No feature cache found")

print("\n" + "="*70)
print("AUDIT COMPLETE")
print("="*70)
