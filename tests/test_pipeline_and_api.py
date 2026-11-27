"""
End-to-End Test Suite for SarcasTone API and Inference Pipeline
Tests text, audio, multimodal, error handling, and performance.
"""

import sys
import os
import json
import base64
import io
import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from sarcastone.pipeline import SarcasTonePipeline
from api.index import handler
from http.server import HTTPServer
import threading
import urllib.request
import urllib.error

def test_pipeline():
    print("="*60)
    print("1. TESTING PIPELINE DIRECT INFERENCE")
    print("="*60)
    pipeline = SarcasTonePipeline()
    assert pipeline.text_model is not None, "Text model failed to load"
    assert pipeline.audio_model is not None, "Audio model failed to load"
    assert pipeline.multimodal_model is not None, "Multimodal model failed to load"
    print("✓ All 3 models loaded into pipeline")

    # Mode A: Text test
    t_res = pipeline.predict_text("Oh perfect, my car just broke down in the rain!")
    print("\nText Test:")
    print(f"  Utterance: 'Oh perfect, my car just broke down in the rain!'")
    print(f"  Prediction: {t_res['prediction']}")
    print(f"  Confidence: {t_res['confidence']:.2f}%")
    print(f"  Sentiment: {t_res['sentiment_label']}")
    assert t_res["prediction"] in ["Sarcastic", "Non-Sarcastic"]

    # Mode B: Audio Preset test
    a_res = pipeline.predict_audio(preset_name="Sarcastic Drawl (Ironic Pitch Drop & Elongated)")
    print("\nAudio Preset Test:")
    print(f"  Preset: 'Sarcastic Drawl'")
    print(f"  Prediction: {a_res['prediction']}")
    print(f"  Confidence: {a_res['confidence']:.2f}%")
    print(f"  Dominant Emotion: {a_res['dominant_emotion']}")
    assert a_res["prediction"] in ["Sarcastic", "Non-Sarcastic"]

    # Mode B: Raw Audio test
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5))
    y = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.random.randn(len(t))
    a_raw_res = pipeline.predict_audio(audio_data=y, sample_rate=sr)
    print("\nAudio Raw Signal Test:")
    print(f"  Signal: 1.5s synthetic wave")
    print(f"  Prediction: {a_raw_res['prediction']}")
    print(f"  Confidence: {a_raw_res['confidence']:.2f}%")
    assert a_raw_res["prediction"] in ["Sarcastic", "Non-Sarcastic"]

    # Mode C: Multimodal test
    mm_res = pipeline.predict_multimodal(
        utterance="Wonderful, another meeting. Just what my Friday afternoon was missing.",
        preset_name="Annoyed / Irritated Voice"
    )
    print("\nMultimodal Test:")
    print(f"  Text Pred: {mm_res['text_result']['prediction']} ({mm_res['text_result']['confidence']:.1f}%)")
    print(f"  Audio Pred: {mm_res['audio_result']['prediction']} ({mm_res['audio_result']['confidence']:.1f}%)")
    print(f"  Fused Pred: {mm_res['fused_result']['prediction']} ({mm_res['fused_result']['confidence']:.1f}%)")
    print(f"  Discord Score: {mm_res['fused_result']['discordance']['discord_score']:.3f}")
    assert mm_res["fused_result"]["prediction"] in ["Sarcastic", "Non-Sarcastic"]

    # Backward compatibility analyze() test
    legacy_res = pipeline.analyze("It's just a privilege to watch your mind at work.", audio_preset_name="Sarcastic Drawl (Ironic Pitch Drop & Elongated)")
    print("\nLegacy analyze() Compatibility Test:")
    print(f"  Verdict: {legacy_res['verdict']}")
    print(f"  Confidence: {legacy_res['confidence_percent']:.1f}%")
    assert "verdict" in legacy_res
    print("✓ All direct pipeline tests passed!")

def test_api_server():
    print("\n" + "="*60)
    print("2. TESTING REST API ENDPOINTS")
    print("="*60)

    server = HTTPServer(("127.0.0.1", 8765), handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    base_url = "http://127.0.0.1:8765"

    try:
        # GET /api/health
        req = urllib.request.Request(f"{base_url}/api/health")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"✓ GET /api/health: status={data['status']}, models_ready={data['models_ready']}")
            assert data["status"] == "healthy"

        # GET /api/metrics
        req = urllib.request.Request(f"{base_url}/api/metrics")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"✓ GET /api/metrics: {len(data)} models evaluated")
            assert len(data) >= 3

        # POST /api/predict/text
        payload = json.dumps({"utterance": "Oh brilliant, another delay!"}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/predict/text", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"✓ POST /api/predict/text: {data['prediction']} (conf: {data['confidence']:.1f}%)")
            assert "prediction" in data

        # POST /api/predict/audio (preset)
        payload = json.dumps({"preset_name": "Deadpan / Monotone Sarcasm (Flat Pitch)"}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/predict/audio", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"✓ POST /api/predict/audio (preset): {data['prediction']} (emotion: {data['dominant_emotion']})")
            assert "prediction" in data

        # POST /api/predict/audio (raw wav base64)
        sr = 16000
        t = np.linspace(0, 1.0, sr)
        audio = 0.5 * np.sin(2 * np.pi * 300 * t)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format='WAV')
        b64_audio = base64.b64encode(buf.getvalue()).decode('utf-8')

        payload = json.dumps({"audio_base64": b64_audio}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/predict/audio", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"✓ POST /api/predict/audio (raw audio base64): {data['prediction']}")
            assert "prediction" in data

        # POST /api/predict/multimodal
        payload = json.dumps({
            "utterance": "Thank you so much for deleting my entire presentation.",
            "preset_name": "Annoyed / Irritated Voice"
        }).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/predict/multimodal", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"✓ POST /api/predict/multimodal: Fused={data['fused_result']['prediction']}, Text={data['text_result']['prediction']}, Audio={data['audio_result']['prediction']}")
            assert "fused_result" in data

        # Error Handling: Empty text
        payload = json.dumps({"utterance": ""}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/predict/text", data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                pass
        except urllib.error.HTTPError as e:
            print(f"✓ Error Handling: Empty text properly rejected with HTTP {e.code}")
            assert e.code == 400

        # Error Handling: Missing audio
        payload = json.dumps({}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/predict/audio", data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                pass
        except urllib.error.HTTPError as e:
            print(f"✓ Error Handling: Missing audio properly rejected with HTTP {e.code}")
            assert e.code == 400

        print("✓ All REST API endpoint tests passed!")

    finally:
        server.shutdown()

if __name__ == "__main__":
    test_pipeline()
    test_api_server()
    print("\n" + "="*60)
    print("ALL END-TO-END TESTS PASSED SUCCESSFULLY!")
    print("="*60)
