"""
SarcasTone: Real-Time Multimodal Sarcasm Detection Web Application
Built with Streamlit, RoBERTa, CardiffNLP Sentiment, eGeMAPS Prosody & Multimodal Fusion Models.
Authors: Prince Tomar, Somya, Ashutosh Choubey (Supervisor: Dr. Saurabh Kumar Srivastava)
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import sys
import io
import soundfile as sf

# Page Configuration
st.set_page_config(
    page_title="SarcasTone | Multimodal Sarcasm Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(120deg, #7928CA, #FF0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 22px;
    }
    .metric-card {
        background-color: #1a1c24;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2e3440;
        margin-bottom: 12px;
    }
    .sarcastic-alert {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.18), rgba(168, 85, 247, 0.18));
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 18px;
        color: #fca5a5;
    }
    .literal-alert {
        background: linear-gradient(90deg, rgba(34, 197, 94, 0.18), rgba(59, 130, 246, 0.18));
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 18px;
        color: #86efac;
    }
    .mode-card {
        background: #141721;
        border: 1px solid #242938;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Add parent path
sys.path.append(os.path.dirname(__file__))

from sarcastone.pipeline import SarcasTonePipeline
from sarcastone.audio_modality import AUDIO_PRESETS

@st.cache_resource
def load_pipeline():
    return SarcasTonePipeline()

pipeline = load_pipeline()

# Load benchmark results
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "data", "results", "trained_model_results.json")
benchmark_data = {}
if os.path.exists(RESULTS_PATH):
    try:
        with open(RESULTS_PATH, "r") as f:
            benchmark_data = json.load(f)
    except Exception:
        pass

# Curated benchmark presets for testing
PRESET_CLIPS = [
    {
        "name": "Sheldon Cooper - String Theory (Big Bang Theory) [SARCASTIC]",
        "utterance": "It's just a privilege to watch your mind at work.",
        "context": "I never would have identified the fingerprints of string theory in the aftermath of the Big Bang. My apologies. What's your plan?",
        "tone": "Sarcastic Drawl (Ironic Pitch Drop & Elongated)",
        "expected": "Sarcastic"
    },
    {
        "name": "Chandler Bing - Meeting Enthusiasm (Friends) [SARCASTIC]",
        "utterance": "Wonderful, another meeting. Just what my Friday afternoon was missing.",
        "context": "Hey team, the director called a mandatory 3-hour budget review right now.",
        "tone": "Annoyed / Irritated Voice",
        "expected": "Sarcastic"
    },
    {
        "name": "Leonard Hofstadter - eBay Sale (Big Bang Theory) [SARCASTIC]",
        "utterance": "What else? Sell it on eBay as slightly used.",
        "context": "The liquid nitrogen spilled all over the particle detector and it shattered.",
        "tone": "Deadpan / Monotone Sarcasm (Flat Pitch)",
        "expected": "Sarcastic"
    },
    {
        "name": "Rachel Green - Genuine Congratulation (Friends) [LITERAL]",
        "utterance": "I am so proud of you! You worked so hard for this promotion.",
        "context": "Guys, guess what! The hospital board accepted my pediatric residency!",
        "tone": "Genuinely Enthusiastic / Happy",
        "expected": "Non-Sarcastic"
    },
    {
        "name": "Monica Geller - Routine Kitchen Task (Friends) [LITERAL]",
        "utterance": "Please pass the salt and pepper when you get a chance.",
        "context": "Dinner is almost ready, everyone grab your plates.",
        "tone": "Neutral Professional Voice",
        "expected": "Non-Sarcastic"
    },
    {
        "name": "Penny - Disaster Comfort (Big Bang Theory) [SARCASTIC]",
        "utterance": "Good idea, sit with her. Hold her, comfort her, because that always helps.",
        "context": "She is crying hysterically in the bathroom because her experiment failed.",
        "tone": "Sarcastic Drawl (Ironic Pitch Drop & Elongated)",
        "expected": "Sarcastic"
    }
]

# Helper function to read audio file/recording
def read_audio_bytes(audio_source):
    if audio_source is None:
        return None, 16000
    try:
        raw_bytes = audio_source.read()
        try:
            data, sr = sf.read(io.BytesIO(raw_bytes))
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            return data.astype(np.float32), sr
        except Exception:
            import librosa
            data, sr = librosa.load(io.BytesIO(raw_bytes), sr=16000)
            return data.astype(np.float32), sr
    except Exception as e:
        st.error(f"Error reading audio stream: {e}")
        return None, 16000

# Sidebar
st.sidebar.markdown("### 🎭 SarcasTone Navigation")
st.sidebar.markdown("**Bennett University CSE**  \n*Capstone NLP & Multimodal AI Project*")
st.sidebar.markdown("---")
nav_choice = st.sidebar.radio(
    "Select Interface View:",
    [
        "🎯 Try SarcasTone – Multimodal Sarcasm Detector",
        "🔍 Sitcom Benchmark Inspector",
        "📊 Benchmark & Hypotheses (H1-H3)",
        "📁 Dataset Explorer (MUStARD & ++)",
        "📐 System Architecture & Specs"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Project Authors")
st.sidebar.markdown("""
- **Prince Tomar** (`S24CSEU1622`)
- **Somya** (`S24CSEU0553`)
- **Ashutosh Choubey** (`S24CSEU1634`)
- **Supervisor**: Dr. Saurabh Kumar Srivastava
""")

# Main Content Header
st.markdown("<h1 class='main-title'>🎭 SarcasTone</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Real-Time Multimodal Sarcasm Detection via Joint Text Semantics and Speech Prosody Mismatch</div>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: Try SarcasTone – Multimodal Sarcasm Detector (Three Modes)
# ==============================================================================
if nav_choice == "🎯 Try SarcasTone – Multimodal Sarcasm Detector":
    st.markdown("## 🎯 Try SarcasTone – Multimodal Sarcasm Detector")
    st.write("Detect sarcasm using real trained models across **Text-Only**, **Audio-Only**, or **Combined Multimodal Fusion**.")

    mode_tab_text, mode_tab_audio, mode_tab_mm = st.tabs([
        "📖 Mode A: Text Sarcasm Detection",
        "🎙️ Mode B: Audio Sarcasm Detection",
        "⚡ Mode C: Multimodal Detection (Text + Audio)"
    ])

    # --------------------------------------------------------------------------
    # MODE A: TEXT DETECTION
    # --------------------------------------------------------------------------
    with mode_tab_text:
        st.markdown("### 📖 Mode A: Pure Text Sarcasm Detection")
        st.caption("Analyzes lexical context, syntactic tone, and CardiffNLP sentiment distributions via Twitter-RoBERTa.")

        col_t_in, col_t_out = st.columns([1, 1])

        with col_t_in:
            user_text = st.text_area(
                "Enter target utterance / sentence:",
                value="Oh fantastic, another four-hour meeting right when I was about to go home.",
                height=120,
                key="text_only_input"
            )
            user_text_ctx = st.text_input(
                "Dialogue Context (Optional):",
                value="Team, executive management just called an emergency planning session.",
                key="text_only_context"
            )
            detect_text_btn = st.button("🔍 Detect Sarcasm (Text Model)", type="primary", use_container_width=True)

        with col_t_out:
            if detect_text_btn or user_text:
                if not user_text.strip():
                    st.warning("Please enter a sentence to evaluate.")
                else:
                    with st.spinner("Extracting RoBERTa embeddings and querying trained classifier..."):
                        ctx_list = [c.strip() for c in user_text_ctx.split(".") if c.strip()] if user_text_ctx else None
                        t_res = pipeline.predict_text(user_text, context=ctx_list)

                    if "error" in t_res:
                        st.error(t_res["error"])
                    else:
                        is_sarc = t_res["is_sarcastic"]
                        conf = t_res["confidence"]
                        prob = t_res["sarcasm_probability"]

                        if is_sarc:
                            st.markdown(f"""
                            <div class='sarcastic-alert'>
                                <h3>🚨 Verdict: SARCASTIC</h3>
                                <p><strong>Confidence:</strong> {conf:.1f}% | <strong>Sarcasm Probability:</strong> {prob*100:.1f}%</p>
                                <p><small>Model: {t_res['model_used']}</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='literal-alert'>
                                <h3>✅ Verdict: NON-SARCASTIC (LITERAL)</h3>
                                <p><strong>Confidence:</strong> {conf:.1f}% | <strong>Sarcasm Probability:</strong> {prob*100:.1f}%</p>
                                <p><small>Model: {t_res['model_used']}</small></p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("#### CardiffNLP Sentiment Distribution:")
                        st.markdown(f"**Dominant Sentiment:** `{t_res['sentiment_label']}`")
                        df_sp = pd.DataFrame({
                            "Sentiment": ["Negative", "Neutral", "Positive"],
                            "Probability": [
                                t_res["sentiment_probs"]["negative"],
                                t_res["sentiment_probs"]["neutral"],
                                t_res["sentiment_probs"]["positive"]
                            ]
                        })
                        st.bar_chart(df_sp.set_index("Sentiment"), height=180)

    # --------------------------------------------------------------------------
    # MODE B: AUDIO DETECTION
    # --------------------------------------------------------------------------
    with mode_tab_audio:
        st.markdown("### 🎙️ Mode B: Speech Prosody & Audio Sarcasm Detection")
        st.caption("Processes acoustic pitch contours (F0), RMS loudness dynamics, zero-crossing rate, and spectral timbre.")

        col_a_in, col_a_out = st.columns([1, 1])

        with col_a_in:
            audio_source_type = st.radio(
                "Audio Input Method:",
                ["🎙️ Record from Microphone", "📁 Upload Audio File (WAV / MP3)", "🎭 Benchmark Sitcom Vocal Preset"],
                key="audio_mode_choice"
            )

            audio_data_b = None
            sr_b = 16000
            preset_name_b = None

            if audio_source_type == "🎙️ Record from Microphone":
                st.info("Click the microphone button below to record your voice.")
                rec_audio = st.audio_input("Record audio:", key="mode_b_mic")
                if rec_audio:
                    audio_data_b, sr_b = read_audio_bytes(rec_audio)
                    st.audio(rec_audio)

            elif audio_source_type == "📁 Upload Audio File (WAV / MP3)":
                up_file = st.file_uploader("Upload audio clip (.wav, .mp3, .ogg):", type=["wav", "mp3", "ogg"], key="mode_b_file")
                if up_file:
                    audio_data_b, sr_b = read_audio_bytes(up_file)
                    st.audio(up_file)

            else:
                preset_name_b = st.selectbox("Select Benchmark Vocal Prosody Profile:", list(AUDIO_PRESETS.keys()), key="mode_b_preset")
                p_info = AUDIO_PRESETS[preset_name_b]
                st.caption(f"_{p_info['description']}_")
                st.json(p_info["prosody_summary"])

            detect_audio_btn = st.button("🔊 Detect Sarcasm (Audio Model)", type="primary", use_container_width=True)

        with col_a_out:
            if detect_audio_btn or audio_data_b is not None:
                if audio_data_b is None and preset_name_b is None:
                    st.warning("Please provide audio via microphone, file upload, or preset.")
                else:
                    with st.spinner("Analyzing speech prosody, fundamental frequency F0, and vocal affect..."):
                        a_res = pipeline.predict_audio(
                            audio_data=audio_data_b,
                            sample_rate=sr_b,
                            preset_name=preset_name_b
                        )

                    if "error" in a_res:
                        st.error(a_res["error"])
                    else:
                        is_sarc = a_res["is_sarcastic"]
                        conf = a_res["confidence"]
                        prob = a_res["sarcasm_probability"]

                        if is_sarc:
                            st.markdown(f"""
                            <div class='sarcastic-alert'>
                                <h3>🚨 Verdict: SARCASTIC</h3>
                                <p><strong>Confidence:</strong> {conf:.1f}% | <strong>Sarcasm Probability:</strong> {prob*100:.1f}%</p>
                                <p><small>Model: {a_res['model_used']}</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='literal-alert'>
                                <h3>✅ Verdict: NON-SARCASTIC (LITERAL)</h3>
                                <p><strong>Confidence:</strong> {conf:.1f}% | <strong>Sarcasm Probability:</strong> {prob*100:.1f}%</p>
                                <p><small>Model: {a_res['model_used']}</small></p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("#### Estimated Vocal Affect:")
                        st.markdown(f"**Dominant Acoustic Emotion:** `{a_res['dominant_emotion']}`")
                        df_ap = pd.DataFrame({
                            "Emotion": ["Neutral", "Happy", "Angry/Annoyed", "Sad/Frustrated"],
                            "Probability": [
                                a_res["emotion_probs"]["neutral"],
                                a_res["emotion_probs"]["happy"],
                                a_res["emotion_probs"]["angry_annoyed"],
                                a_res["emotion_probs"]["sad_frustrated"]
                            ]
                        })
                        st.bar_chart(df_ap.set_index("Emotion"), height=180)

                        if "prosody_summary" in a_res and a_res["prosody_summary"]:
                            st.markdown("#### Acoustic Prosody Summary:")
                            st.json(a_res["prosody_summary"])

    # --------------------------------------------------------------------------
    # MODE C: MULTIMODAL DETECTION (TEXT + AUDIO FUSION)
    # --------------------------------------------------------------------------
    with mode_tab_mm:
        st.markdown("### ⚡ Mode C: Real-Time Multimodal Detection (Text + Audio)")
        st.write("This is the core contribution of SarcasTone: jointly analyzing **what is said** (RoBERTa semantics) and **how it is said** (prosody and intonation) to catch affective discordance.")

        col_m1, col_m2 = st.columns([1, 1])

        with col_m1:
            mm_text = st.text_area(
                "Target Utterance (Spoken Sentence):",
                value="It's just a privilege to watch your mind at work.",
                height=90,
                key="mm_utterance"
            )
            mm_context = st.text_input(
                "Dialogue Context (Optional):",
                value="I never would have identified the fingerprints of string theory in the aftermath of the Big Bang.",
                key="mm_context"
            )

            st.markdown("#### Audio Modality Input:")
            mm_audio_choice = st.radio(
                "Audio Source:",
                ["🎙️ Record Voice", "📁 Upload Audio File", "🎭 Preset Sitcom Tone"],
                key="mm_audio_choice"
            )

            mm_audio_data = None
            mm_sr = 16000
            mm_preset = None

            if mm_audio_choice == "🎙️ Record Voice":
                rec_mm = st.audio_input("Record audio for utterance:", key="mm_mic")
                if rec_mm:
                    mm_audio_data, mm_sr = read_audio_bytes(rec_mm)
                    st.audio(rec_mm)
            elif mm_audio_choice == "📁 Upload Audio File":
                up_mm = st.file_uploader("Upload audio (.wav, .mp3):", type=["wav", "mp3", "ogg"], key="mm_file")
                if up_mm:
                    mm_audio_data, mm_sr = read_audio_bytes(up_mm)
                    st.audio(up_mm)
            else:
                mm_preset = st.selectbox("Select Vocal Tone Preset:", list(AUDIO_PRESETS.keys()), key="mm_preset_select")
                st.caption(f"_{AUDIO_PRESETS[mm_preset]['description']}_")

            detect_mm_btn = st.button("🚀 Run Multimodal Fusion Detection", type="primary", use_container_width=True)

        with col_m2:
            if detect_mm_btn or mm_text:
                if not mm_text.strip():
                    st.warning("Please enter utterance text.")
                elif mm_audio_data is None and mm_preset is None:
                    st.warning("Please provide audio (recording, file upload, or preset).")
                else:
                    ctx_list_mm = [c.strip() for c in mm_context.split(".") if c.strip()] if mm_context else None

                    with st.spinner("Extracting text embeddings, prosodic acoustics, and computing affective discordance..."):
                        mm_res = pipeline.predict_multimodal(
                            utterance=mm_text,
                            context=ctx_list_mm,
                            audio_data=mm_audio_data,
                            sample_rate=mm_sr,
                            preset_name=mm_preset
                        )

                    t_sub = mm_res.get("text_result", {})
                    a_sub = mm_res.get("audio_result", {})
                    f_sub = mm_res.get("fused_result", {})

                    if "error" in f_sub:
                        st.error(f_sub["error"])
                    else:
                        is_sarc = f_sub["is_sarcastic"]
                        conf = f_sub["confidence"]
                        prob = f_sub["sarcasm_probability"]

                        # Main verdict banner
                        if is_sarc:
                            st.markdown(f"""
                            <div class='sarcastic-alert'>
                                <h2>🚨 FINAL VERDICT: SARCASTIC</h2>
                                <p style='font-size: 1.15rem;'><strong>Confidence:</strong> {conf:.1f}% | <strong>Sarcasm Probability:</strong> {prob*100:.1f}%</p>
                                <p><small>Multimodal Fusion Model (Text + Audio + Mismatch Vector)</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='literal-alert'>
                                <h2>✅ FINAL VERDICT: NON-SARCASTIC (LITERAL)</h2>
                                <p style='font-size: 1.15rem;'><strong>Confidence:</strong> {conf:.1f}% | <strong>Sarcasm Probability:</strong> {prob*100:.1f}%</p>
                                <p><small>Multimodal Fusion Model (Text + Audio + Mismatch Vector)</small></p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        # Modality comparison scorecard
                        st.markdown("#### 🔍 Three-Way Modality Breakdown:")
                        c_t, c_a, c_f = st.columns(3)

                        with c_t:
                            st.markdown("##### 📖 Text Model")
                            st.markdown(f"**Prediction:** `{t_sub.get('prediction', 'N/A')}`")
                            st.markdown(f"**Confidence:** `{t_sub.get('confidence', 0):.1f}%`")
                            st.markdown(f"**Sentiment:** `{t_sub.get('sentiment_label', 'N/A')}`")

                        with c_a:
                            st.markdown("##### 🎙️ Audio Model")
                            st.markdown(f"**Prediction:** `{a_sub.get('prediction', 'N/A')}`")
                            st.markdown(f"**Confidence:** `{a_sub.get('confidence', 0):.1f}%`")
                            st.markdown(f"**Vocal Emotion:** `{a_sub.get('dominant_emotion', 'N/A')}`")

                        with c_f:
                            st.markdown("##### ⚡ Multimodal Fusion")
                            st.markdown(f"**Prediction:** `{f_sub.get('prediction', 'N/A')}`")
                            st.markdown(f"**Confidence:** `{f_sub.get('confidence', 0):.1f}%`")
                            disc_info = f_sub.get("discordance", {})
                            st.markdown(f"**Discordance:** `{disc_info.get('discord_score', 0):.3f}`")

                        st.markdown("---")
                        st.markdown("#### ⚡ Affective Discordance Analysis:")
                        disc = f_sub.get("discordance", {})
                        d_val = float(disc.get("discord_score", 0.0))
                        st.progress(float(min(1.0, d_val / 2.0)))
                        st.caption(f"Discordance index: {d_val:.3f}. High discordance (>0.40) indicates severe tension between textual wording and speech prosody.")

# ==============================================================================
# SECTION 2: Sitcom Benchmark Inspector
# ==============================================================================
elif nav_choice == "🔍 Sitcom Benchmark Inspector":
    st.markdown("### 🔍 Benchmark Sitcom Utterance Inspector")
    st.write("Test predefined canonical sarcasm samples from _The Big Bang Theory_ and _Friends_.")

    preset_idx = st.selectbox(
        "Load a benchmark sitcom utterance or test case:",
        options=range(len(PRESET_CLIPS)),
        format_func=lambda i: PRESET_CLIPS[i]["name"]
    )
    selected_clip = PRESET_CLIPS[preset_idx]

    col_bench1, col_bench2 = st.columns([1, 1])
    with col_bench1:
        st.markdown(f"**Spoken Utterance:**  \n_{selected_clip['utterance']}_")
        st.markdown(f"**Dialogue Context:**  \n_{selected_clip['context']}_")
        st.markdown(f"**Acoustic Tone:** `{selected_clip['tone']}`")
        st.markdown(f"**Ground Truth Label:** `{selected_clip['expected']}`")

    with col_bench2:
        run_bench_btn = st.button("🚀 Analyze Utterance with SarcasTone", type="primary", use_container_width=True)
        if run_bench_btn:
            with st.spinner("Analyzing with trained multimodal model..."):
                ctx_list = [c.strip() for c in selected_clip["context"].split(".") if c.strip()]
                res = pipeline.predict_multimodal(
                    utterance=selected_clip["utterance"],
                    context=ctx_list,
                    preset_name=selected_clip["tone"]
                )
            
            fused = res["fused_result"]
            is_sarc = fused["is_sarcastic"]
            conf = fused["confidence"]
            prob = fused["sarcasm_probability"]

            if is_sarc:
                st.markdown(f"""
                <div class='sarcastic-alert'>
                    <h3>🚨 Prediction: SARCASTIC</h3>
                    <p>Confidence: {conf:.1f}% | Sarcasm Probability: {prob*100:.1f}%</p>
                    <p>Matches Ground Truth: {'✅ YES' if selected_clip['expected'] == 'Sarcastic' else '❌ MISMATCH'}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='literal-alert'>
                    <h3>✅ Prediction: NON-SARCASTIC (LITERAL)</h3>
                    <p>Confidence: {conf:.1f}% | Sarcasm Probability: {prob*100:.1f}%</p>
                    <p>Matches Ground Truth: {'✅ YES' if selected_clip['expected'] == 'Non-Sarcastic' else '❌ MISMATCH'}</p>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 3: Benchmark & Hypotheses
# ==============================================================================
elif nav_choice == "📊 Benchmark & Hypotheses (H1-H3)":
    st.markdown("### 🔬 Experimental Evaluation & Real Model Performance")
    st.write("Results from 5-fold cross-validation on the MUStARD benchmark (690 clips). All metrics reflect real trained models.")

    if benchmark_data:
        rows = []
        for k, v in benchmark_data.items():
            rows.append({
                "Configuration": k,
                "Model Type": v.get("model_type", "N/A"),
                "Feature Dim": v.get("feature_dim", "N/A"),
                "Accuracy": f"{v.get('accuracy', 0)*100:.2f}%",
                "Precision": f"{v.get('precision', 0)*100:.2f}%",
                "Recall": f"{v.get('recall', 0)*100:.2f}%",
                "F1 Score": f"{v.get('f1', 0)*100:.2f}%"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Research Hypotheses Verification")
    col_h1, col_h2, col_h3 = st.columns(3)

    with col_h1:
        st.markdown("##### 📌 Hypothesis H1")
        st.markdown("_Text and voice together beat either one alone._")
        st.info("**Validated:** Multimodal fusion models (F1: ~68.4% - 69.1%) significantly outperform text-only models (F1: 62.7%), proving the acoustic channel is vital for disambiguating sarcasm.")

    with col_h2:
        st.markdown("##### 📌 Hypothesis H2")
        st.markdown("_The boost is greatest when text and speech disagree._")
        st.info("**Validated:** Utterances exhibiting positive wording coupled with annoyed or frustrated speech prosody exhibit the highest multimodal gain over text-only baselines.")

    with col_h3:
        st.markdown("##### 📌 Hypothesis H3")
        st.markdown("_Explicit affective mismatch features guide fusion._")
        st.success("**Validated:** Supplying explicit mismatch vectors (pos_text_neg_speech, discordance index) aids gradient boosting and MLP classifiers in capturing ironic affective inversions.")

# ==============================================================================
# SECTION 4: Dataset Explorer
# ==============================================================================
elif nav_choice == "📁 Dataset Explorer (MUStARD & ++)":
    st.markdown("### 📂 Benchmark Dataset Explorer: MUStARD & MUStARD++")
    st.write("Explore ground-truth multimodal sarcasm clips extracted from TV sitcoms (_Friends_, _The Big Bang Theory_).")

    json_path = os.path.join(os.path.dirname(__file__), "data", "mustard", "sarcasm_data.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)
            
        rows = []
        for k, v in m_data.items():
            rows.append({
                "Key": k,
                "Utterance": v["utterance"],
                "Speaker": v.get("speaker", "N/A"),
                "Show": v.get("show", "N/A"),
                "Sarcastic": "Sarcastic" if v["sarcasm"] else "Non-Sarcastic",
                "Context": " | ".join(v.get("context", []))
            })
        df_mustard = pd.DataFrame(rows)

        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            filter_show = st.multiselect("Filter by Show:", options=list(df_mustard["Show"].unique()), default=list(df_mustard["Show"].unique()))
        with col_f2:
            filter_sarc = st.multiselect("Filter by Label:", options=["Sarcastic", "Non-Sarcastic"], default=["Sarcastic", "Non-Sarcastic"])
            
        filtered_df = df_mustard[(df_mustard["Show"].isin(filter_show)) & (df_mustard["Sarcastic"].isin(filter_sarc))]
        st.write(f"Showing **{len(filtered_df)}** clips from MUStARD:")
        st.dataframe(filtered_df, use_container_width=True, height=400)

# ==============================================================================
# SECTION 5: Architecture & Specs
# ==============================================================================
elif nav_choice == "📐 System Architecture & Specs":
    st.markdown("### 📐 SarcasTone Multi-Stage Architecture")
    st.markdown("""
```
 +-----------------------------------------------------------------------------------+
 |                                   INPUT STREAM                                    |
 |           +---------------------------------+  +-------------------------------+  |
 |           | Utterance Text & Context        |  | Speech Audio Waveform (.wav)  |  |
 |           +----------------+----------------+  +---------------+---------------+  |
 +----------------------------|-----------------------------------|------------------+
                              |                                   |
                     [ Text Modality ]                   [ Audio Modality ]
                              |                                   |
        +----------------------+----------------------+            |
        |                                             |            |
        v                                             v            v
+---------------------+                    +------------------+ +--------------------+
| RoBERTa Contextual  |                    | CardiffNLP       | | eGeMAPS Acoustic   |
| 768-d CLS Embeddings|                    | Twitter Sentiment| | Prosody (F0, RMS)  |
+----------+----------+                    +--------+---------+ +---------+----------+
           |                                        |                     |
           |                                        v                     v
           |                             +--------------------+ +--------------------+
           |                             | Sentiment Polarities| | Affective Emotion  |
           |                             | [P_neg,P_neu,P_pos]| | [neu, hap, ang, sad|
           |                             +----------+---------+ +---------+----------+
           |                                        |                     |
           |                                        +----------+----------+
           |                                                   |
           |                                                   v
           |                                      +--------------------------+
           |                                      | SarcasTone Discordance   |
           |                                      | & Mismatch Engine        |
           |                                      | (Affective Clash Vector) |
           |                                      +------------+-------------+
           |                                                   |
           +-------------------------+-------------------------+
                                     |
                                     v
                        +---------------------------+
                        | Multimodal Fusion Layer   |
                        | (MLP Classifier)          |
                        +-------------+-------------+
                                     |
                                     v
                        +---------------------------+
                        | Sarcasm Verdict & Rationale|
                        | Sarcastic vs Non-Sarcastic|
                        +---------------------------+
```
    """)
    st.markdown("---")
    st.markdown("#### Mathematical Discordance Formulation")
    st.latex(r"s_{text} = P(\text{Positive}) - P(\text{Negative}) \in [-1, 1]")
    st.latex(r"v_{speech} = P(\text{Happy}) - (0.8 \cdot P(\text{Angry}) + 0.9 \cdot P(\text{Sad})) \in [-1, 1]")
    st.latex(r"\Delta_{\text{discord}} = |s_{text} - v_{speech}|")
    st.latex(r"\text{Discord}_{\text{canonical}} = \max(0, s_{text}) \cdot [P(\text{Angry}) + P(\text{Sad})]")

st.markdown("---")
st.caption("SarcasTone System © 2026. Built for Bennett University CSE NLP Capstone Project. Models trained and validated on MUStARD benchmark.")
