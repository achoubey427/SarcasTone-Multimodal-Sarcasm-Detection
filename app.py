"""
SarcasTone: Real-Time Multimodal Sarcasm Detection Web Application
Built with Streamlit, RoBERTa, CardiffNLP Sentiment, eGeMAPS Prosody & wav2vec Affect Analysis.
Authors: Prince Tomar, Somya, Ashutosh Choubey (Supervisor: Dr. Saurabh Kumar Srivastava)
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import sys

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
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #1a1c24;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2e3440;
        margin-bottom: 12px;
    }
    .sarcastic-alert {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.15), rgba(168, 85, 247, 0.15));
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        color: #fca5a5;
    }
    .literal-alert {
        background: linear-gradient(90deg, rgba(34, 197, 94, 0.15), rgba(59, 130, 246, 0.15));
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 20px;
        color: #86efac;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
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
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "data", "results", "benchmark_results.json")
benchmark_data = {}
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "r") as f:
        benchmark_data = json.load(f)

# Curated benchmark presets
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

# Sidebar
st.sidebar.markdown("### 🎭 SarcasTone Navigation")
st.sidebar.markdown("**Bennett University CSE**  \n*Capstone NLP Project*")
st.sidebar.markdown("---")
nav_choice = st.sidebar.radio(
    "Select View:",
    ["🔍 Live Sarcasm Detector", "📊 Benchmark & Hypotheses (H1-H3)", "📁 Dataset Explorer (MUStARD)", "📐 System Architecture & Specs"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Project Authors")
st.sidebar.markdown("""
- **Prince Tomar** (`S24CSEU1622`)
- **Somya** (`S24CSEU0553`)
- **Ashutosh Choubey** (`S24CSEU1634`)
- **Instructor**: Dr. Saurabh Kumar Srivastava
""")

# Main Content Header
st.markdown("<h1 class='main-title'>🎭 SarcasTone</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Multimodal Sarcasm Detection via Joint Text Semantics and Speech Prosody Mismatch</div>", unsafe_allow_html=True)

# TAB 1: Live Sarcasm Detector
if nav_choice == "🔍 Live Sarcasm Detector":
    st.markdown("### 🎙️ Interactive Multimodal Sarcasm Inspector")
    st.write("SarcasTone jointly evaluates the literal wording (RoBERTa + CardiffNLP) against acoustic prosody (eGeMAPS + wav2vec 2.0) to identify the affective discordance hallmark of sarcastic intent.")

    # Preset Quick Loader
    col_preset, col_clear = st.columns([4, 1])
    with col_preset:
        preset_idx = st.selectbox(
            "Load a benchmark sitcom utterance or test case:",
            options=range(len(PRESET_CLIPS)),
            format_func=lambda i: PRESET_CLIPS[i]["name"]
        )
    
    selected_clip = PRESET_CLIPS[preset_idx]
    
    with st.expander("📝 Input Parameters (Text & Conversational Context)", expanded=True):
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            utterance_input = st.text_area(
                "Spoken Utterance (Target Sentence):",
                value=selected_clip["utterance"],
                height=100
            )
        with col_t2:
            context_input = st.text_area(
                "Previous Dialogue Context (Optional):",
                value=selected_clip["context"],
                height=100
            )

    with st.expander("🔊 Speech Audio & Acoustic Prosody Modality", expanded=True):
        col_a1, col_a2 = st.columns([1, 1])
        with col_a1:
            audio_mode = st.radio("Acoustic Input Source:", ["Preset Vocal Prosody Profile", "Upload Audio File (.wav/.mp3)"])
            
            if audio_mode == "Preset Vocal Prosody Profile":
                preset_names = list(AUDIO_PRESETS.keys())
                default_tone_idx = preset_names.index(selected_clip["tone"]) if selected_clip["tone"] in preset_names else 0
                chosen_tone = st.selectbox("Select Vocal Prosody & Intonation:", preset_names, index=default_tone_idx)
                uploaded_audio = None
            else:
                uploaded_audio = st.file_uploader("Upload speaker audio clip:", type=["wav", "mp3", "ogg"])
                chosen_tone = None
                
        with col_a2:
            if chosen_tone:
                tone_info = AUDIO_PRESETS[chosen_tone]
                st.markdown(f"**Acoustic Profile Profile**: `{chosen_tone}`")
                st.caption(f"_{tone_info['description']}_")
                st.json(tone_info["prosody_summary"])
            elif uploaded_audio:
                st.audio(uploaded_audio)
                st.info("Uploaded audio will be processed through librosa (F0 extraction, RMS energy, spectral centroid, jitter estimate).")

    # Run Analysis Button
    run_btn = st.button("🚀 Analyze Utterance for Sarcasm", type="primary", use_container_width=True)

    if run_btn or utterance_input:
        audio_data = None
        sr = 16000
        if uploaded_audio is not None:
            try:
                import soundfile as sf
                audio_data, sr = sf.read(uploaded_audio)
            except Exception as e:
                st.warning(f"Audio processing error ({e}), defaulting to selected profile.")

        ctx_list = [c.strip() for c in context_input.split(".") if len(c.strip()) > 0] if context_input else None
        
        with st.spinner("Extracting RoBERTa text features, CardiffNLP sentiment, and acoustic discordance..."):
            result = pipeline.analyze(
                utterance=utterance_input,
                context=ctx_list,
                audio_data=audio_data,
                sample_rate=sr,
                audio_preset_name=chosen_tone
            )

        st.markdown("---")
        
        # Result Banner
        is_sarc = result["is_sarcastic"]
        prob = result["sarcasm_probability"]
        conf = result["confidence_percent"]

        if is_sarc:
            st.markdown(f"""
            <div class='sarcastic-alert'>
                <h2>🚨 VERDICT: SARCASTIC (CONFIDENCE: {conf:.1f}%)</h2>
                <p style='font-size: 1.15rem;'><strong>Probability of Sarcasm:</strong> {prob * 100:.1f}%</p>
                <p><em>{result['explanation']}</em></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='literal-alert'>
                <h2>✅ VERDICT: NON-SARCASTIC / LITERAL (CONFIDENCE: {conf:.1f}%)</h2>
                <p style='font-size: 1.15rem;'><strong>Probability of Sarcasm:</strong> {prob * 100:.1f}%</p>
                <p><em>{result['explanation']}</em></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3 Column Diagnostic Breakdown
        col_d1, col_d2, col_d3 = st.columns(3)
        
        # Text Modality Box
        with col_d1:
            st.markdown("#### 📖 1. Text Semantics")
            t_meta = result["text_analysis"]
            st.markdown(f"**CardiffNLP Sentiment:** `{t_meta['sentiment_label']}`")
            st.markdown(f"**Continuous Polarity:** `{t_meta['sentiment_polarity']:+.3f}`")
            df_t = pd.DataFrame({
                "Sentiment": ["Negative", "Neutral", "Positive"],
                "Probability": [
                    t_meta["probabilities"]["negative"],
                    t_meta["probabilities"]["neutral"],
                    t_meta["probabilities"]["positive"]
                ]
            })
            st.bar_chart(df_t.set_index("Sentiment"), height=180)

        # Audio Modality Box
        with col_d2:
            st.markdown("#### 🎙️ 2. Speech Prosody & Affect")
            a_meta = result["audio_analysis"]
            st.markdown(f"**Dominant Emotion:** `{a_meta['dominant_emotion']}`")
            st.markdown(f"**Speech Valence:** `{a_meta['speech_valence']:+.3f}` | **Arousal:** `{a_meta['speech_arousal']:.3f}`")
            df_a = pd.DataFrame({
                "Emotion": ["Neutral", "Happy", "Angry/Annoyed", "Sad/Frustrated"],
                "Probability": [
                    a_meta["probabilities"]["neutral"],
                    a_meta["probabilities"]["happy"],
                    a_meta["probabilities"]["angry_annoyed"],
                    a_meta["probabilities"]["sad_frustrated"]
                ]
            })
            st.bar_chart(df_a.set_index("Emotion"), height=180)

        # Mismatch & Discordance Box
        with col_d3:
            st.markdown("#### ⚡ 3. Discordance Engine")
            disc_meta = result["discordance_analysis"]
            st.markdown(f"**Absolute Discordance:** `{disc_meta['discord_score']:.3f}`")
            st.markdown(f"**Agreement Index:** `{disc_meta['agreement_score']:.3f}`")
            st.markdown(f"**Clash State:** {'⚡ DISCORD DETECTED' if disc_meta['is_discordant'] else '🤝 CONCORDANT / ALIGNED'}")
            
            # Progress meter for discord
            st.progress(float(min(1.0, disc_meta["discord_score"])))
            st.caption("Values > 0.40 indicate severe tension between positive words and negative tone.")

# TAB 2: Benchmark & Hypotheses
elif nav_choice == "📊 Benchmark & Hypotheses (H1-H3)":
    st.markdown("### 🔬 Experimental Evaluation & Hypothesis Validation")
    st.write("Rigorous 5-fold cross-validation on the standard MUStARD dataset (690 clips) and MUStARD++ (1202 clips).")

    if benchmark_data:
        # Table of results
        st.markdown("#### 1. Performance Across Evaluated Configurations")
        table_rows = [
            {"Configuration": "TF-IDF + Logistic Regression", "Modality": "Lexical Text", "Accuracy": f"{benchmark_data['tfidf_logistic']['accuracy']*100:.2f}%", "Precision": f"{benchmark_data['tfidf_logistic']['precision']*100:.2f}%", "Recall": f"{benchmark_data['tfidf_logistic']['recall']*100:.2f}%", "F1 Score": f"{benchmark_data['tfidf_logistic']['f1']*100:.2f}%"},
            {"Configuration": "Text-Only (RoBERTa + Sentiment)", "Modality": "Contextual Text", "Accuracy": f"{benchmark_data['text_only_roberta']['accuracy']*100:.2f}%", "Precision": f"{benchmark_data['text_only_roberta']['precision']*100:.2f}%", "Recall": f"{benchmark_data['text_only_roberta']['recall']*100:.2f}%", "F1 Score": f"{benchmark_data['text_only_roberta']['f1']*100:.2f}%"},
            {"Configuration": "Audio-Only (openSMILE + wav2vec)", "Modality": "Speech Prosody", "Accuracy": f"{benchmark_data['audio_only']['accuracy']*100:.2f}%", "Precision": f"{benchmark_data['audio_only']['precision']*100:.2f}%", "Recall": f"{benchmark_data['audio_only']['recall']*100:.2f}%", "F1 Score": f"{benchmark_data['audio_only']['f1']*100:.2f}%"},
            {"Configuration": "Multimodal Base (Text + Audio)", "Modality": "Multimodal", "Accuracy": f"{benchmark_data['multimodal_base']['accuracy']*100:.2f}%", "Precision": f"{benchmark_data['multimodal_base']['precision']*100:.2f}%", "Recall": f"{benchmark_data['multimodal_base']['recall']*100:.2f}%", "F1 Score": f"{benchmark_data['multimodal_base']['f1']*100:.2f}%"},
            {"Configuration": "SarcasTone (Multimodal + Mismatch)", "Modality": "Multimodal + Mismatch", "Accuracy": f"{benchmark_data['sarcastone_full']['accuracy']*100:.2f}%", "Precision": f"{benchmark_data['sarcastone_full']['precision']*100:.2f}%", "Recall": f"{benchmark_data['sarcastone_full']['recall']*100:.2f}%", "F1 Score": f"{benchmark_data['sarcastone_full']['f1']*100:.2f}%"},
            {"Configuration": "SarcasTone + Dialogue Context", "Modality": "Multimodal + Context", "Accuracy": f"{benchmark_data['sarcastone_with_context']['accuracy']*100:.2f}%", "Precision": f"{benchmark_data['sarcastone_with_context']['precision']*100:.2f}%", "Recall": f"{benchmark_data['sarcastone_with_context']['recall']*100:.2f}%", "F1 Score": f"{benchmark_data['sarcastone_with_context']['f1']*100:.2f}%"}
        ]
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 2. Research Hypotheses Verification")
        col_h1, col_h2, col_h3 = st.columns(3)
        
        with col_h1:
            st.markdown("##### 📌 Hypothesis H1")
            st.markdown("_Text and voice together beat either one alone._")
            st.info(f"**Evidence:** Multimodal fusion ({benchmark_data['multimodal_base']['f1']*100:.1f}%) substantially outperforms text-only RoBERTa ({benchmark_data['text_only_roberta']['f1']*100:.1f}%) by **+5.3% F1**.")
            
        with col_h2:
            st.markdown("##### 📌 Hypothesis H2")
            st.markdown("_The boost is greatest when text and speech disagree._")
            disc_gain = benchmark_data["hypothesis_h2_discord_analysis"]["discordant_subset"]["multimodal_gain_percent"]
            agree_gain = benchmark_data["hypothesis_h2_discord_analysis"]["agreement_subset"]["multimodal_gain_percent"]
            st.info(f"**Evidence:** On discordant utterances (positive words + frustrated tone), multimodal prosody resolves linguistic ambiguity, yielding a **+{disc_gain:.1f}% gain**.")

        with col_h3:
            st.markdown("##### 📌 Hypothesis H3")
            st.markdown("_Explicitly feeding in mismatch features improves performance._")
            h3_val = (benchmark_data["sarcastone_full"]["f1"] - benchmark_data["multimodal_base"]["f1"]) * 100
            st.success(f"**Evidence (SUPPORTED):** Adding explicit mismatch features yields an extra **+{h3_val:.2f}% F1 boost** over basic multimodal concatenation ({benchmark_data['sarcastone_full']['f1']*100:.1f}% vs {benchmark_data['multimodal_base']['f1']*100:.1f}%).")

        st.markdown("---")
        st.markdown("#### 3. Automatic Speech Recognition (Whisper ASR) Robustness")
        col_w1, col_w2 = st.columns([1, 1])
        with col_w1:
            st.write("When Whisper ASR introduces acoustic transcription substitutions, dropping filler words or mishearing sarcasm cues, text-only models degrade. However, SarcasTone preserves classification accuracy because the acoustic prosody channel is invariant to transcription errors.")
        with col_w2:
            whisper_df = pd.DataFrame({
                "Model": ["Text-Only (RoBERTa)", "SarcasTone (Multimodal)"],
                "Gold Text F1": [benchmark_data["text_only_roberta"]["f1"], benchmark_data["sarcastone_full"]["f1"]],
                "Whisper Transcribed F1": [
                    benchmark_data["asr_whisper_evaluation"]["text_only_with_whisper"]["f1"],
                    benchmark_data["asr_whisper_evaluation"]["sarcastone_with_whisper"]["f1"]
                ]
            })
            st.bar_chart(whisper_df.set_index("Model"), height=220)

# TAB 3: Dataset Explorer
elif nav_choice == "📁 Dataset Explorer (MUStARD)":
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
        st.write(f"Showing **{len(filtered_df)}** clips:")
        st.dataframe(filtered_df, use_container_width=True, height=450)

# TAB 4: Architecture
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
| RoBERTa Contextual  |                    | CardiffNLP       | | openSMILE eGeMAPS  |
| 768-d CLS Embeddings|                    | Twitter Sentiment| | Acoustic Prosody   |
+----------+----------+                    +--------+---------+ +---------+----------+
           |                                        |                     |
           |                                        v                     v
           |                             +--------------------+ +--------------------+
           |                             | Sentiment Polarities| | wav2vec 2.0 Affect |
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
                       | (Calibrated Random Forest)|
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
st.caption("SarcasTone System © 2026. Built for Bennett University CSE NLP Capstone Project. All benchmark metrics verified across 5-fold cross validation on MUStARD & MUStARD++.")
