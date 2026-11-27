"""
SarcasTone: Production Backend REST API Server / Serverless Handler
Provides endpoints for text, audio, and multimodal sarcasm detection.
Maintains cached model instances for high throughput and sub-second latency.
"""

from http.server import BaseHTTPRequestHandler
import json
import urllib.parse
import sys
import os
import io
import base64
import numpy as np

# Ensure root workspace directory is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Global pipeline singleton
_PIPELINE = None

def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        try:
            from sarcastone.pipeline import SarcasTonePipeline
            _PIPELINE = SarcasTonePipeline()
            print("✓ SarcasTonePipeline singleton successfully initialized for API.")
        except Exception as e:
            print(f"✗ Failed to initialize SarcasTonePipeline: {e}")
            _PIPELINE = None
    return _PIPELINE

# Load trained results metadata if available
RESULTS_CACHE = {}
results_file = os.path.join(ROOT_DIR, "data", "results", "trained_model_results.json")
if os.path.exists(results_file):
    try:
        with open(results_file, "r") as f:
            RESULTS_CACHE = json.load(f)
    except Exception:
        pass

def parse_audio_data(audio_bytes, filename=None):
    """
    Decodes audio bytes into float32 numpy array and sample rate.
    Supports WAV, MP3, OGG, FLAC.
    """
    import soundfile as sf
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32), sr
    except Exception:
        # Fallback to librosa
        try:
            import librosa
            data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            return data.astype(np.float32), sr
        except Exception as e:
            raise ValueError(f"Unable to parse audio stream: {e}")

class handler(BaseHTTPRequestHandler):
    """Vercel / standard HTTP Serverless Handler."""

    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("", "/api", "/api/health", "/api/status"):
            pipeline = get_pipeline()
            models_loaded = pipeline is not None and (
                pipeline.text_model is not None and 
                pipeline.audio_model is not None and 
                pipeline.multimodal_model is not None
            )
            response = {
                "status": "healthy" if models_loaded else "degraded",
                "service": "SarcasTone Multimodal Sarcasm Detection API",
                "version": "2.0.0",
                "models_ready": models_loaded,
                "endpoints": {
                    "predict_text": "POST /api/predict/text",
                    "predict_audio": "POST /api/predict/audio",
                    "predict_multimodal": "POST /api/predict/multimodal",
                    "benchmark_metrics": "GET /api/metrics",
                    "dataset_audit": "GET /api/dataset"
                }
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))
            return

        if path == "/api/metrics":
            self._set_headers(200)
            self.wfile.write(json.dumps(RESULTS_CACHE, indent=2).encode("utf-8"))
            return

        if path == "/api/dataset":
            audit_summary = {
                "mustard": {
                    "total_samples": 690,
                    "sarcastic": 345,
                    "non_sarcastic": 345,
                    "text_available": True,
                    "pre_extracted_audio_matrices": 690,
                    "raw_audio_files": 0,
                    "status": "Partially Multimodal (Trained with aligned acoustic frames)"
                },
                "mustard_plus": {
                    "total_labeled": 1202,
                    "sarcastic": 601,
                    "non_sarcastic": 601,
                    "text_available": True,
                    "raw_audio_files": 0,
                    "status": "Text-Only"
                }
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(audit_summary, indent=2).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": f"Endpoint not found: {path}"}).encode("utf-8"))

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        pipeline = get_pipeline()

        if pipeline is None:
            self._set_headers(503)
            self.wfile.write(json.dumps({"error": "SarcasTone pipeline not ready or failed to load."}).encode("utf-8"))
            return

        # Read content length
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 15 * 1024 * 1024:  # 15MB limit
            self._set_headers(413)
            self.wfile.write(json.dumps({"error": "Payload exceeds 15MB limit."}).encode("utf-8"))
            return

        body = self.rfile.read(content_length)

        # -------------------------------------------------------------
        # Mode A: Text Sarcasm Detection
        # -------------------------------------------------------------
        if path == "/api/predict/text":
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Malformed JSON payload."}).encode("utf-8"))
                return

            utterance = data.get("utterance", "").strip()
            if not utterance:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Field 'utterance' is required and cannot be empty."}).encode("utf-8"))
                return

            context = data.get("context", None)
            if isinstance(context, str):
                context = [c.strip() for c in context.split(".") if c.strip()]

            res = pipeline.predict_text(utterance, context=context)
            self._set_headers(200)
            self.wfile.write(json.dumps(res, indent=2).encode("utf-8"))
            return

        # -------------------------------------------------------------
        # Mode B: Audio Sarcasm Detection
        # -------------------------------------------------------------
        if path == "/api/predict/audio":
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Malformed JSON payload."}).encode("utf-8"))
                return

            preset_name = data.get("preset_name")
            audio_b64 = data.get("audio_base64")

            if not preset_name and not audio_b64:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Provide either 'audio_base64' (raw audio) or 'preset_name'."}).encode("utf-8"))
                return

            audio_data = None
            sample_rate = 16000
            if audio_b64:
                try:
                    raw_bytes = base64.b64decode(audio_b64)
                    audio_data, sample_rate = parse_audio_data(raw_bytes)
                except Exception as e:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": f"Invalid audio stream: {str(e)}"}).encode("utf-8"))
                    return

            res = pipeline.predict_audio(audio_data=audio_data, sample_rate=sample_rate, preset_name=preset_name)
            self._set_headers(200)
            self.wfile.write(json.dumps(res, indent=2).encode("utf-8"))
            return

        # -------------------------------------------------------------
        # Mode C: Multimodal Detection (Text + Audio Fusion)
        # -------------------------------------------------------------
        if path == "/api/predict/multimodal":
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Malformed JSON payload."}).encode("utf-8"))
                return

            utterance = data.get("utterance", "").strip()
            preset_name = data.get("preset_name")
            audio_b64 = data.get("audio_base64")
            context = data.get("context", None)
            if isinstance(context, str):
                context = [c.strip() for c in context.split(".") if c.strip()]

            if not utterance:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Field 'utterance' is required for multimodal analysis."}).encode("utf-8"))
                return

            if not preset_name and not audio_b64:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Audio input ('preset_name' or 'audio_base64') is required for multimodal analysis."}).encode("utf-8"))
                return

            audio_data = None
            sample_rate = 16000
            if audio_b64:
                try:
                    raw_bytes = base64.b64decode(audio_b64)
                    audio_data, sample_rate = parse_audio_data(raw_bytes)
                except Exception as e:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": f"Invalid audio stream: {str(e)}"}).encode("utf-8"))
                    return

            res = pipeline.predict_multimodal(
                utterance=utterance,
                context=context,
                audio_data=audio_data,
                sample_rate=sample_rate,
                preset_name=preset_name
            )
            self._set_headers(200)
            self.wfile.write(json.dumps(res, indent=2).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": f"Endpoint not found: {path}"}).encode("utf-8"))

if __name__ == "__main__":
    from http.server import HTTPServer
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), handler)
    print(f"Starting SarcasTone API server on port {port}...")
    server.serve_forever()
