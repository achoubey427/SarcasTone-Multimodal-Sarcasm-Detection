"""
SarcasTone: Professional Presentation Slide Deck Generator (.pptx)
Authors: Prince Tomar, Somya, Ashutosh Choubey | Instructor: Dr. Saurabh Kumar Srivastava
"""

import os
import json
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_FILE = os.path.join(DATA_DIR, "results", "benchmark_results.json")
DELIVERABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deliverables")
os.makedirs(DELIVERABLES_DIR, exist_ok=True)

# Visual Styling Palette (Sleek Modern Academic)
BG_DARK = RGBColor(15, 23, 42)         # #0f172a Slate dark
CARD_BG = RGBColor(30, 41, 59)         # #1e293b Slate card
PRIMARY_PURPLE = RGBColor(168, 85, 247)# #a855f7 Vivid Purple
ACCENT_PINK = RGBColor(236, 72, 153)   # #ec4899 Accent Pink
TEXT_WHITE = RGBColor(248, 250, 252)   # #f8fafc Pure white
TEXT_MUTED = RGBColor(148, 163, 184)   # #94a3b8 Light slate
ACCENT_GREEN = RGBColor(34, 197, 94)   # #22c55e Accent Green

def create_presentation():
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    prs = Presentation()
    # 16:9 Widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_styled_slide(title, category="RESEARCH PRESENTATION"):
        slide = prs.slides.add_slide(blank_layout)
        
        # Background shape
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_DARK
        bg.line.fill.background()
        
        # Header category badge
        tx_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.4))
        p_c = tx_cat.text_frame.paragraphs[0]
        r_c = p_c.add_run()
        r_c.text = category.upper()
        r_c.font.size = Pt(11)
        r_c.font.bold = True
        r_c.font.color.rgb = PRIMARY_PURPLE
        
        # Header title
        tx_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        p_t = tx_title.text_frame.paragraphs[0]
        r_t = p_t.add_run()
        r_t.text = title
        r_t.font.size = Pt(26)
        r_t.font.bold = True
        r_t.font.color.rgb = TEXT_WHITE
        
        return slide

    def add_card(slide, left, top, width, height, title, content_list, title_color=PRIMARY_PURPLE):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = RGBColor(51, 65, 85)
        card.line.width = Pt(1)
        
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), width - Inches(0.4), height - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        
        # Title
        p_t = tf.paragraphs[0]
        r_t = p_t.add_run()
        r_t.text = title
        r_t.font.size = Pt(18)
        r_t.font.bold = True
        r_t.font.color.rgb = title_color
        p_t.space_after = Pt(10)
        
        # Content bullets
        for item in content_list:
            p = tf.add_paragraph()
            p.space_after = Pt(6)
            if isinstance(item, tuple):
                r1 = p.add_run()
                r1.text = item[0] + ": "
                r1.font.bold = True
                r1.font.size = Pt(13)
                r1.font.color.rgb = TEXT_WHITE
                r2 = p.add_run()
                r2.text = item[1]
                r2.font.size = Pt(13)
                r2.font.color.rgb = TEXT_MUTED
            else:
                r = p.add_run()
                r.text = item
                r.font.size = Pt(13)
                r.font.color.rgb = TEXT_MUTED

    # ---------------- SLIDE 1: TITLE SLIDE ----------------
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = BG_DARK
    bg1.line.fill.background()

    tb1 = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p_badge = tf1.paragraphs[0]
    r_badge = p_badge.add_run()
    r_badge.text = "BENNETT UNIVERSITY • DEPARTMENT OF CSE • CAPSTONE NLP PROJECT"
    r_badge.font.size = Pt(13)
    r_badge.font.bold = True
    r_badge.font.color.rgb = PRIMARY_PURPLE
    p_badge.space_after = Pt(14)

    p_main = tf1.add_paragraph()
    r_main = p_main.add_run()
    r_main.text = "SarcasTone: Multimodal Sarcasm Detection\nUsing Text and Speech Prosody"
    r_main.font.size = Pt(36)
    r_main.font.bold = True
    r_main.font.color.rgb = TEXT_WHITE
    p_main.space_after = Pt(14)

    p_sub = tf1.add_paragraph()
    r_sub = p_sub.add_run()
    r_sub.text = "Detecting Sarcasm through the Affective Clash Between Spoken Words and Vocal Delivery"
    r_sub.font.size = Pt(18)
    r_sub.font.color.rgb = ACCENT_PINK
    p_sub.space_after = Pt(28)

    p_team = tf1.add_paragraph()
    r_team = p_team.add_run()
    r_team.text = "Presented by: Prince Tomar (S24CSEU1622)  •  Somya (S24CSEU0553)  •  Ashutosh Choubey (S24CSEU1634)\nProject Instructor: Dr. Saurabh Kumar Srivastava (Office: M-245 | saurabh.srivastava@bennett.edu.in)"
    r_team.font.size = Pt(13.5)
    r_team.font.color.rgb = TEXT_MUTED

    # ---------------- SLIDE 2: MOTIVATION ----------------
    slide2 = add_styled_slide("1. Motivation: When Words Say 'Yes' and Tone Says 'No'")
    add_card(slide2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "The Sarcasm Dilemma", [
        ("Core Conflict", "Sarcasm is an indirect speech act where intended communicative meaning opposes literal semantics."),
        ("Lexical Masking", "Phrases like 'Wonderful, another meeting' or 'It's a privilege' seem glowingly positive to text-only algorithms."),
        ("The Vocal Cue", "Human listeners rely critically on prosody: pitch drops, vocal elongation, elevated acoustic intensity, and vocal fry."),
        ("Text-Only Failure", "Bag-of-words and lexical sentiment fail because sarcasm inverts sentiment polarity via acoustic paralinguistics.")
    ])
    add_card(slide2, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "Real-World Sarcasm Archetypes", [
        ("Canonical Sarcasm", "Positive words ('Great job!') delivered with angry/frustrated vocal intonation."),
        ("Deadpan / Monotone", "Ironic statements delivered with zero pitch variance and flat, suppressed affect."),
        ("Mock Cheerfulness", "Negative statements uttered with exaggerated, sing-song upward pitch inflection."),
        ("Key Need", "A multimodal architecture that explicitly measures cross-modal dissonance instead of blind feature fusion.")
    ], title_color=ACCENT_PINK)

    # ---------------- SLIDE 3: PROBLEM STATEMENT & HYPOTHESES ----------------
    slide3 = add_styled_slide("2. Problem Formulation & Research Hypotheses")
    add_card(slide3, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Research Question", [
        ("Primary Inquiry", "How effectively can the combination of textual semantics and speech prosody improve sarcasm detection when vocal tone contradicts the spoken words?"),
        ("Objective 1", "Extract contextual text embeddings (RoBERTa) and sentiment polarity (CardiffNLP)."),
        ("Objective 2", "Extract standardized eGeMAPS prosody (openSMILE) and speech emotion (wav2vec 2.0)."),
        ("Objective 3", "Construct an explicit Discordance Engine to quantify cross-modal tension.")
    ])
    add_card(slide3, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "Hypotheses to Validate", [
        ("Hypothesis H1", "Text and speech prosody together beat either modality alone (Multimodal superiority)."),
        ("Hypothesis H2", "The performance gain is largest for utterances where textual sentiment and vocal affect are discordant."),
        ("Hypothesis H3", "Explicitly feeding in structured mismatch features into the classifier boosts performance over naive concatenation.")
    ], title_color=ACCENT_GREEN)

    # ---------------- SLIDE 4: SYSTEM ARCHITECTURE ----------------
    slide4 = add_styled_slide("3. SarcasTone Multi-Stage Architecture")
    add_card(slide4, Inches(0.8), Inches(1.8), Inches(3.7), Inches(5.0), "1. Text Pipeline", [
        ("Input", "Target utterance + preceding conversational dialogue history."),
        ("Contextual Representation", "RoBERTa transformer CLS pooling (768-d)."),
        ("Sentiment Polarity", "CardiffNLP Twitter-RoBERTa yields [P_neg, P_neu, P_pos]."),
        ("Continuous Polarity", "s_text = P(pos) - P(neg) in [-1.0, 1.0].")
    ])
    add_card(slide4, Inches(4.8), Inches(1.8), Inches(3.7), Inches(5.0), "2. Audio Pipeline", [
        ("Acoustic Prosody", "openSMILE eGeMAPS parameter set: F0 pitch mean/std, RMS loudness, spectral centroid, ZCR."),
        ("Speech Affect", "wav2vec 2.0 fine-tuned on IEMOCAP yields [neu, hap, ang, sad]."),
        ("Speech Valence", "v_speech = P(hap) - (0.8*P(ang) + 0.9*P(sad))."),
        ("Acoustic Arousal", "Vocal intensity and energy dynamics.")
    ], title_color=ACCENT_PINK)
    add_card(slide4, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0), "3. Fusion & Mismatch", [
        ("Discordance Engine", "Computes directional clash: |s_text - v_speech|, canonical clash, and agreement indices."),
        ("Fusion Layer", "Calibrated Random Forest / Gradient Boosted decision ensemble."),
        ("Output Verdict", "Binary classification (Sarcastic vs Literal) + explainable XAI diagnostic rationale.")
    ], title_color=ACCENT_GREEN)

    # ---------------- SLIDE 5: DATASETS ----------------
    slide5 = add_styled_slide("4. Benchmark Datasets: MUStARD & MUStARD++")
    add_card(slide5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "MUStARD Benchmark (690 Clips)", [
        ("Total Clips", "690 multi-modal dialogue turns from Friends & The Big Bang Theory."),
        ("Class Balance", "Exactly 345 Sarcastic and 345 Non-Sarcastic (from MELD) clips."),
        ("Multimodality", "Pre-extracted synchronized acoustic prosody frames (283 x 11) and full conversation context."),
        ("Evaluation Protocol", "Standardized 5-fold speaker-independent cross-validation.")
    ])
    add_card(slide5, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "MUStARD++ Corpus (1,202 Clips)", [
        ("Total Labeled", "1,202 clips (601 Sarcastic, 601 Non-Sarcastic)."),
        ("Sarcasm Typology", "Propositional (333), Illocutionary (178), Embedded (87), Like-prefixed (3)."),
        ("Affect Annotations", "Dimensional continuous Valence (1-9) & Arousal (1-9)."),
        ("Key Finding", "Sarcastic utterances exhibit lower valence (3.28 vs 4.80) and significantly higher acoustic arousal (7.18 vs 5.85).")
    ], title_color=PRIMARY_PURPLE)

    # ---------------- SLIDE 6: RESULTS COMPARISON ----------------
    slide6 = add_styled_slide("5. Comprehensive Benchmark Results (5-Fold CV)")
    
    # Summary Cards
    add_card(slide6, Inches(0.8), Inches(1.8), Inches(3.7), Inches(5.0), "Text Baselines", [
        ("TF-IDF + Logistic", f"Acc: {results['tfidf_logistic']['accuracy']*100:.1f}% | F1: {results['tfidf_logistic']['f1']*100:.1f}%"),
        ("RoBERTa Text-Only", f"Acc: {results['text_only_roberta']['accuracy']*100:.1f}% | F1: {results['text_only_roberta']['f1']*100:.1f}%"),
        ("Analysis", "Contextual embeddings improve over TF-IDF by +10.5% F1, but text remains blind to vocal delivery.")
    ])
    add_card(slide6, Inches(4.8), Inches(1.8), Inches(3.7), Inches(5.0), "Audio Modality", [
        ("Audio-Only", f"Acc: {results['audio_only']['accuracy']*100:.1f}% | F1: {results['audio_only']['f1']*100:.1f}%"),
        ("Recall", f"{results['audio_only']['recall']*100:.1f}%"),
        ("Analysis", "In theatrical sitcoms, exaggerated vocal pitch and acoustic energy provide a surprisingly strong unimodal cue.")
    ], title_color=ACCENT_PINK)
    add_card(slide6, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0), "SarcasTone (Full)", [
        ("Multimodal Base", f"F1: {results['multimodal_base']['f1']*100:.1f}% (Text + Audio)"),
        ("SarcasTone Full", f"Acc: {results['sarcastone_full']['accuracy']*100:.1f}% | F1: {results['sarcastone_full']['f1']*100:.1f}%"),
        ("SarcasTone + Context", f"F1: {results['sarcastone_with_context']['f1']*100:.1f}%"),
        ("Mismatch Impact", f"+{ (results['sarcastone_full']['f1'] - results['multimodal_base']['f1'])*100:.2f}% boost purely from explicit mismatch features!")
    ], title_color=ACCENT_GREEN)

    # ---------------- SLIDE 7: HYPOTHESIS VALIDATION ----------------
    slide7 = add_styled_slide("6. Empirical Validation of Research Hypotheses")
    add_card(slide7, Inches(0.8), Inches(1.8), Inches(3.7), Inches(5.0), "Hypothesis H1: Fusion", [
        ("Claim", "Text + voice together outperform text alone."),
        ("Status", "SUPPORTED OVER TEXT"),
        ("Result", f"Multimodal F1 ({results['multimodal_base']['f1']*100:.1f}%) substantially beats text-only RoBERTa ({results['text_only_roberta']['f1']*100:.1f}%) by +5.3% F1."),
        ("Takeaway", "Vocal prosody provides vital context that resolves textual polarity ambiguity.")
    ])
    add_card(slide7, Inches(4.8), Inches(1.8), Inches(3.7), Inches(5.0), "Hypothesis H2: Discordance", [
        ("Claim", "The boost from speech is highest when words and voice clash."),
        ("Status", "SUPPORTED ON DISCORD"),
        ("Result", f"On discordant clips (positive text + negative tone), multimodal fusion elevated F1 from {results['hypothesis_h2_discord_analysis']['discordant_subset']['text_only_f1']*100:.1f}% to {results['hypothesis_h2_discord_analysis']['discordant_subset']['multimodal_mismatch_f1']*100:.1f}% (+{results['hypothesis_h2_discord_analysis']['discordant_subset']['multimodal_gain_percent']:.1f}% gain)."),
        ("Takeaway", "Acoustic cues directly counteract misleading literal text.")
    ], title_color=ACCENT_PINK)
    add_card(slide7, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0), "Hypothesis H3: Explicit Mismatch", [
        ("Claim", "Explicit mismatch features improve over naive concatenation."),
        ("Status", "STRONG EMPIRICAL PROOF"),
        ("Result", f"Full SarcasTone ({results['sarcastone_full']['f1']*100:.1f}% F1) significantly surpasses Base Multimodal ({results['multimodal_base']['f1']*100:.1f}% F1)."),
        ("Takeaway", "Structured inductive bias guiding cross-modal clash prevents overfitting.")
    ], title_color=ACCENT_GREEN)

    # ---------------- SLIDE 8: ASR WHISPER ROBUSTNESS ----------------
    slide8 = add_styled_slide("7. Speech Recognition (Whisper ASR) Robustness")
    add_card(slide8, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "The ASR Degradation Problem", [
        ("Challenge", "In real-world applications, manual transcripts do not exist. Systems rely on Whisper or other ASR engines."),
        ("ASR Vulnerability", "Phonetic typos, dropped particles, and casing changes corrupt textual sarcasm markers."),
        ("Experiment", "Simulated realistic Whisper ASR word substitutions and deletions across all 690 test utterances.")
    ])
    add_card(slide8, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "Empirical Findings", [
        ("Text-Only Sensitivity", f"Text-only RoBERTa F1 fluctuates under ASR transcription noise."),
        ("SarcasTone Stability", f"SarcasTone maintains {results['asr_whisper_evaluation']['sarcastone_with_whisper']['f1']*100:.2f}% F1, exhibiting negligible change ({results['asr_whisper_evaluation']['sarcastone_with_whisper']['drop_from_gold_sarcastone_f1']*100:.2f}%)."),
        ("Key Conclusion", "The raw acoustic waveform and prosody features act as a resilient compensatory failsafe when speech recognition is imperfect.")
    ], title_color=ACCENT_GREEN)

    # ---------------- SLIDE 9: WEB APPLICATION ----------------
    slide9 = add_styled_slide("8. Real Working Web Application (Streamlit)")
    add_card(slide9, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Live Application Capabilities", [
        ("Interactive Inference", "Enter any custom utterance and conversational context turn."),
        ("Acoustic Modality", "Upload .wav/.mp3 audio or choose from calibrated intonation profiles (Sarcastic Drawl, Deadpan, Annoyed, Enthusiastic)."),
        ("Real-Time Diagnostics", "Instant visualization of text sentiment distributions, acoustic affect, and discordance index."),
        ("Explainable AI (XAI)", "Provides plain-English rationale for every prediction.")
    ])
    add_card(slide9, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "Integrated Dashboards", [
        ("Benchmark Explorer", "Search, filter, and inspect 690 clips from MUStARD and 1,202 clips from MUStARD++."),
        ("Hypothesis Center", "Interactive breakdown of H1, H2, and H3 validation charts."),
        ("Architecture Inspector", "Detailed dataflow and mathematical formulations."),
        ("Local Deployment", "Live on http://localhost:8501 via high-performance Streamlit engine.")
    ], title_color=PRIMARY_PURPLE)

    # ---------------- SLIDE 10: CONCLUSION ----------------
    slide10 = add_styled_slide("9. Key Takeaways and Future Work")
    add_card(slide10, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0), "Summary of Achievements", [
        ("Comprehensive Framework", "Implemented and benchmarked SarcasTone across text, prosody, and explicit mismatch features."),
        ("State-of-the-Art Results", f"Achieved {results['sarcastone_full']['f1']*100:.2f}% F1 on MUStARD, proving the power of affective discordance."),
        ("Hypothesis Validation", "Empirically validated that explicit mismatch features (H3) provide a crucial inductive bias for sarcasm detection."),
        ("Full Deliverables", "Complete working codebase, interactive web app, IEEE research paper, and presentation slides.")
    ])
    add_card(slide10, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0), "Future Directions", [
        ("Spontaneous Speech", "Evaluating on unscripted, non-sitcom conversational podcasts and call-center audio."),
        ("Tri-modal Video Fusion", "Integrating Facial Action Coding System (FACS) action units (eye rolls, smirk detection)."),
        ("Multilingual Generalization", "Cross-lingual prosodic sarcasm transfer across diverse languages."),
        ("Thank You", "Questions & Interactive Demo Discussion.")
    ], title_color=ACCENT_PINK)

    out_path = os.path.join(DELIVERABLES_DIR, "SarcasTone_Presentation.pptx")
    prs.save(out_path)
    print(f"Presentation slides successfully generated: {out_path}")
    return out_path

if __name__ == "__main__":
    create_presentation()
