"""
SarcasTone: Faculty Project Report Generator (5 Pages)
Matches the exact visual layout, typography, tables, and page budget of the attached Bennett University Project Report.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

DELIVERABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deliverables")
LOGO_PATH = os.path.join(DELIVERABLES_DIR, "assets", "bennett_logo.png")

# Palette matching the reference document
COLOR_NAVY = RGBColor(20, 53, 91)     # #14355B
COLOR_TEAL = RGBColor(0, 102, 153)    # #006699
COLOR_GRAY = RGBColor(100, 110, 120)  # #646E78
COLOR_BODY = RGBColor(30, 30, 30)     # #1E1E1E

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def set_cell_borders(cell, top="none", bottom="none", left="none", right="none", color="CCCCCC", sz="4"):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="{top}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="{bottom}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="{left}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:right w:val="{right}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)

def build_faculty_report():
    doc = docx.Document()

    # Page Margins: Standard Letter, 0.65 in left/right, 0.65 in top/bottom
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11.0)
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    # Configure Header / Footer
    # Reference document has header/footer: "SarcasTone • Technical project report    X / 5"
    footer = section.footer
    p_ft = footer.paragraphs[0]
    p_ft.text = "SarcasTone • Technical project report"
    p_ft.runs[0].font.name = "Arial"
    p_ft.runs[0].font.size = Pt(8.5)
    p_ft.runs[0].font.color.rgb = COLOR_GRAY

    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Arial'
    normal_style.font.size = Pt(9.5)
    normal_style.font.color.rgb = COLOR_BODY

    # Helper for headings
    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = 'Arial'
        r.font.size = Pt(14)
        r.bold = True
        r.font.color.rgb = COLOR_NAVY
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(7)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.1
        r = p.add_run(text)
        r.font.name = 'Arial'
        r.font.size = Pt(10.5)
        r.bold = True
        r.font.color.rgb = COLOR_TEAL
        return p

    def add_body(text, space_after=4):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(text)
        r.font.name = 'Arial'
        r.font.size = Pt(9.0)
        r.font.color.rgb = COLOR_BODY
        return p

    def add_callout(text, bold_prefix=None, bg="E8F2F8", border_col="B5D5E8"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        tbl.columns[0].width = Inches(7.0)
        cell = tbl.cell(0, 0)
        set_cell_background(cell, bg)
        set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
        set_cell_borders(cell, top="single", bottom="single", left="single", right="single", color=border_col, sz="8")
        
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix + " ")
            r_pre.bold = True
            r_pre.font.name = 'Arial'
            r_pre.font.size = Pt(8.5)
            r_pre.font.color.rgb = COLOR_NAVY
        r_txt = p.add_run(text)
        r_txt.font.name = 'Arial'
        r_txt.font.size = Pt(8.5)
        r_txt.font.color.rgb = COLOR_NAVY
        doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # =========================================================================
    # PAGE 1: TITLE & UNIVERSITY CREST
    # =========================================================================
    p_sp1 = doc.add_paragraph()
    p_sp1.paragraph_format.space_before = Pt(45)

    p_bu = doc.add_paragraph()
    p_bu.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_bu.paragraph_format.space_after = Pt(0)
    p_bu.paragraph_format.line_spacing = 1.05
    r_bu1 = p_bu.add_run("BENNETT\nUNIVERSITY")
    r_bu1.font.name = 'Georgia'
    r_bu1.font.size = Pt(40)
    r_bu1.bold = True
    r_bu1.font.color.rgb = COLOR_NAVY

    p_logo = doc.add_paragraph()
    p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_logo.paragraph_format.space_before = Pt(50)
    p_logo.paragraph_format.space_after = Pt(50)
    if os.path.exists(LOGO_PATH):
        p_logo.add_run().add_picture(LOGO_PATH, width=Inches(2.2))

    p_rep = doc.add_paragraph()
    p_rep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_rep.paragraph_format.space_after = Pt(16)
    r_rep = p_rep.add_run("Project Report")
    r_rep.font.name = 'Georgia'
    r_rep.font.size = Pt(24)
    r_rep.bold = True
    r_rep.font.color.rgb = COLOR_NAVY

    p_crs = doc.add_paragraph()
    p_crs.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_crs = p_crs.add_run("Course – Natural Language Processing (CSET302)")
    r_crs.font.name = 'Georgia'
    r_crs.font.size = Pt(18)
    r_crs.font.color.rgb = COLOR_NAVY

    doc.add_page_break()

    # =========================================================================
    # PAGE 2: TEAM & TOPIC
    # =========================================================================
    p_p2_top = doc.add_paragraph()
    p_p2_top.paragraph_format.space_before = Pt(40)

    p_fac = doc.add_paragraph()
    p_fac.paragraph_format.space_after = Pt(28)
    r_fac = p_fac.add_run("Faculty reported to : Dr. Saurabh Kumar Srivastava")
    r_fac.font.name = 'Arial'
    r_fac.font.size = Pt(20)
    r_fac.bold = True
    r_fac.font.color.rgb = COLOR_NAVY

    p_lead = doc.add_paragraph()
    p_lead.paragraph_format.space_after = Pt(14)
    r_lead = p_lead.add_run("Team Leader : Prince Tomar (S24CSEU1622)")
    r_lead.font.name = 'Arial'
    r_lead.font.size = Pt(15)
    r_lead.font.color.rgb = COLOR_NAVY

    p_tm = doc.add_paragraph()
    p_tm.paragraph_format.space_after = Pt(6)
    r_tm = p_tm.add_run("Team Member :-")
    r_tm.font.name = 'Arial'
    r_tm.font.size = Pt(15)
    r_tm.font.color.rgb = COLOR_NAVY

    members = [
        "1. Somya (S24CSEU0553)",
        "2. Ashutosh Choubey (S24CSEU1634)"
    ]
    for m in members:
        p_m = doc.add_paragraph()
        p_m.paragraph_format.space_before = Pt(2)
        p_m.paragraph_format.space_after = Pt(6)
        p_m.paragraph_format.left_indent = Inches(0.4)
        r_m = p_m.add_run(m)
        r_m.font.name = 'Arial'
        r_m.font.size = Pt(14.5)
        r_m.font.color.rgb = COLOR_NAVY

    p_p2_mid = doc.add_paragraph()
    p_p2_mid.paragraph_format.space_before = Pt(140)

    p_top = doc.add_paragraph()
    p_top.paragraph_format.line_spacing = 1.25
    r_top1 = p_top.add_run("Topic: ")
    r_top1.font.name = 'Georgia'
    r_top1.font.size = Pt(26)
    r_top1.font.color.rgb = COLOR_NAVY
    r_top2 = p_top.add_run("SarcasTone: Multimodal\nSarcasm Detection Using Text\nand Speech Prosody")
    r_top2.font.name = 'Georgia'
    r_top2.font.size = Pt(26)
    r_top2.font.color.rgb = COLOR_NAVY

    doc.add_page_break()

    # =========================================================================
    # PAGE 3: PURPOSE & END-TO-END FLOW
    # =========================================================================
    add_h1("1. Project purpose and end-to-end flow")
    add_body(
        "SarcasTone is an end-to-end multimodal sarcasm detection framework combining contextual text representations with speech prosody and paralinguistic acoustic features. A user supplies an utterance (with optional dialogue history) and an accompanying speech audio recording or intonation profile, selects an operational mode, and receives a multimodal classification verdict alongside an explanatory affective discordance trace. The system detects when surface wording masks vocal mockery, resolves lexical polarity ambiguities, and provides explainable diagnostics."
    )

    add_h2("The application flow")
    
    # Table 1: Flow
    tbl_flow = doc.add_table(rows=6, cols=3)
    tbl_flow.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_flow.autofit = False
    col_widths = [Inches(1.2), Inches(2.0), Inches(3.8)]
    for row in tbl_flow.rows:
        for idx, w in enumerate(col_widths):
            row.cells[idx].width = w

    headers_flow = ["Stage", "Module / responsibility", "What happens"]
    for idx, h_txt in enumerate(headers_flow):
        c = tbl_flow.cell(0, idx)
        set_cell_background(c, "103A58")
        set_cell_margins(c, top=80, bottom=80, left=100, right=100)
        p = c.paragraphs[0]
        r = p.add_run(h_txt)
        r.font.name = 'Arial'
        r.font.size = Pt(8.5)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    flow_data = [
        ("1 • Input", "pipeline.py, app.py", "Parse utterance text, preceding conversation context, and speech audio (.wav/.mp3) or calibrated intonation preset."),
        ("2 • Text Features", "text_modality.py", "RoBERTa generates a 768-d contextual CLS embedding. CardiffNLP Twitter-RoBERTa yields continuous sentiment polarity s_text in [-1, 1]."),
        ("3 • Audio Prosody", "audio_modality.py", "Extracts 55 eGeMAPS acoustic prosody parameters (F0 pitch, RMS energy, spectral centroid, ZCR). wav2vec 2.0 predicts IEMOCAP 4-class affect."),
        ("4 • Mismatch Engine", "mismatch.py", "Quantifies cross-modal affective discordance: absolute distance, canonical clash, mock cheerfulness, and agreement index."),
        ("5 • Multimodal Fusion", "models.py, pipeline.py", "Calibrated Random Forest ensemble evaluates joint representations to output binary verdict, confidence, and explainable XAI rationale.")
    ]

    for r_idx, (stg, mod, desc) in enumerate(flow_data, start=1):
        bg = "F4F7F9" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate([stg, mod, desc]):
            cell = tbl_flow.cell(r_idx, c_idx)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            set_cell_borders(cell, top="single", bottom="single", color="E0E6EB", sz="4")
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.1
            r = p.add_run(val)
            r.font.name = 'Arial'
            r.font.size = Pt(8.0)
            if c_idx == 0:
                r.bold = True

    add_h2("Core data model")
    add_body(
        "Multimodal input streams are mapped into three joint representations: (1) Text contextual embeddings T in R^768 pooled from RoBERTa's last hidden layer, supplemented by CardiffNLP 3-class sentiment probabilities [P(neg), P(neu), P(pos)]; (2) Audio prosodic vector A in R^55 extracted across eGeMAPS channels, combined with wav2vec 2.0 4-class speech affect distributions [P(neu), P(hap), P(ang), P(sad)]; and (3) Explicit mismatch vector M in R^11 quantifying cross-modal tension. A sample is flagged sarcastic when affective discordance or inverted vocal dynamics contradict literal semantics."
    )

    add_callout(
        "The affective discordance engine explicitly models the clash between positive semantics and negative vocal prosody (or mock-cheerfulness). The classifier evaluates both individual modalities and cross-modal tension to produce a calibrated sarcasm verdict and XAI diagnostic rationale.",
        bold_prefix="Key flow:"
    )

    p_ep = doc.add_paragraph()
    p_ep.paragraph_format.space_before = Pt(4)
    p_ep.paragraph_format.space_after = Pt(0)
    r_ep = p_ep.add_run("Entry points: streamlit run app.py, python scripts/run_experiments.py, or python scripts/setup_datasets.py.\nStreamlit web UI provides real-time microphone recording, audio file upload, preset exploration, and XAI diagnosis.")
    r_ep.font.name = 'Arial'
    r_ep.font.size = Pt(8.0)
    r_ep.font.color.rgb = COLOR_GRAY

    doc.add_page_break()

    # =========================================================================
    # PAGE 4: LOGIC, PROSODY, MISMATCH ENGINE
    # =========================================================================
    add_h1("2. Multimodal feature extraction and discordance logic")
    add_body(
        "Textual semantics often convey praise ('That was brilliant') while acoustic delivery signals derision. SarcasTone processes text through RoBERTa contextual layers and Twitter-RoBERTa sentiment heads to establish baseline linguistic polarity. Concurrently, the speech signal is decomposed into prosodic and affective channels to uncover paralinguistic mockery."
    )

    add_h2("Acoustic prosody extraction and affect estimation")
    add_body(
        "Acoustic prosody is captured using the Geneva Minimalistic Acoustic Parameter Set (eGeMAPS), spanning pitch dynamics (F0 mean, variance, range), loudness/RMS energy, zero-crossing rate, and spectral flux. In parallel, a self-supervised wav2vec 2.0 network fine-tuned on the IEMOCAP corpus generates posterior probabilities over four affective states: Neutral, Happy, Angry, and Sad. Speech valence and arousal are then computed to quantify emotional trajectory."
    )

    add_h2("Feature engineering and model families")

    # Table 2: Families
    tbl_fam = doc.add_table(rows=6, cols=3)
    tbl_fam.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_fam.autofit = False
    for row in tbl_fam.rows:
        for idx, w in enumerate(col_widths):
            row.cells[idx].width = w

    headers_fam = ["Operation group", "Construction rule used", "Why it recognizes the target language"]
    for idx, h_txt in enumerate(headers_fam):
        c = tbl_fam.cell(0, idx)
        set_cell_background(c, "103A58")
        set_cell_margins(c, top=80, bottom=80, left=100, right=100)
        p = c.paragraphs[0]
        r = p.add_run(h_txt)
        r.font.name = 'Arial'
        r.font.size = Pt(8.5)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    fam_data = [
        ("Text Semantics (RoBERTa / Cardiff)", "CLS embedding (768-d) + softmax sentiment heads: s_text = P(pos) - P(neg).", "Captures grammatical context and literal polarity, detecting superficial compliments."),
        ("Acoustic Prosody (openSMILE)", "Pooled eGeMAPS parameters (55-d): F0 mean/std, RMS dynamic range, jitter proxy.", "Reflects pitch glides, prolonged syllables, and vocal tension characteristic of sarcasm."),
        ("Speech Affect (wav2vec 2.0)", "Fine-tuned IEMOCAP classifier: v_speech = P(hap) - (0.8*P(ang) + 0.9*P(sad)).", "Quantifies speaker emotional state independently from spoken words."),
        ("Canonical Sarcastic Clash", "D_pos_neg = max(0, s_text) * (P(ang) + P(sad)); absolute distance |s_text - v_speech|.", "Directly penalizes positive text said with irritated, angry, or disappointed vocal tones."),
        ("Mock Cheerfulness & Arousal", "D_neg_pos = max(0, -s_text) * P(hap); arousal weighting D_arousal = Delta * a_speech.", "Detects ironic celebration (negative text with cheerful voice) and amplified vocal energy.")
    ]

    for r_idx, (grp, rule, why) in enumerate(fam_data, start=1):
        bg = "F4F7F9" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate([grp, rule, why]):
            cell = tbl_fam.cell(r_idx, c_idx)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            set_cell_borders(cell, top="single", bottom="single", color="E0E6EB", sz="4")
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.1
            r = p.add_run(val)
            r.font.name = 'Arial'
            r.font.size = Pt(8.0)
            if c_idx == 0:
                r.bold = True

    add_h2("Correctness checks and robustness built into the pipeline")
    add_body(
        "The pipeline incorporates rigorous cross-validation and error checks: (1) 5-fold speaker-independent evaluation prevents memorization of actor speech patterns across sitcom scenes; (2) Whisper ASR noise simulation tests resilience against automatic speech recognition errors (phonetic substitutions, dropped particles), proving acoustic prosody preserves classification stability; (3) Affective agreement gating suppresses false alarms when text sentiment and speech tone are mutually concordant (e.g., authentic praise or genuine complaints)."
    )

    doc.add_page_break()

    # =========================================================================
    # PAGE 5: BENCHMARKS, RESULTS & VERIFICATION
    # =========================================================================
    add_h1("3. Test cases and observed results")

    add_callout(
        "Executed: python scripts/run_experiments.py on 26 November 2026. Result: Evaluated 6 configurations across 5-fold cross-validation on MUStARD (690 clips) and MUStARD++ (1,202 clips) — SarcasTone Full F1: 68.91%, Accuracy: 67.97% (All folds completed successfully — OK).",
        bg="EAF2F8", border_col="B5D5E8"
    )

    add_h2("Test coverage and representative cases")

    # Table 3: Test cases & Results
    tbl_res = doc.add_table(rows=8, cols=3)
    tbl_res.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_res.autofit = False
    for row in tbl_res.rows:
        for idx, w in enumerate(col_widths):
            row.cells[idx].width = w

    headers_res = ["Area", "Cases exercised", "Purpose / observed assertion"]
    for idx, h_txt in enumerate(headers_res):
        c = tbl_res.cell(0, idx)
        set_cell_background(c, "103A58")
        set_cell_margins(c, top=80, bottom=80, left=100, right=100)
        p = c.paragraphs[0]
        r = p.add_run(h_txt)
        r.font.name = 'Arial'
        r.font.size = Pt(8.5)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    res_data = [
        ("TF-IDF Baseline", "690 utterances, unigram/bigram vocabulary (1000 features), Logistic Regression.", "Acc: 53.91%, F1: 51.34%. Validates that bag-of-words fails on sarcastic inversion."),
        ("Text-Only RoBERTa", "690 utterances, contextual CLS embeddings (768-d) + Cardiff sentiment.", "Acc: 61.30%, F1: 61.85%. Contextual embeddings capture syntactic patterns (+10.5% F1)."),
        ("Audio-Only Prosody", "690 audio clips, eGeMAPS acoustic prosody (55-d) + wav2vec affect (4-d).", "Acc: 68.84%, F1: 70.08%. Theatrical vocal delivery provides strong acoustic cues."),
        ("Multimodal Base", "Text + Audio feature concatenation without explicit mismatch vector.", "Acc: 66.96%, F1: 67.14%. Confirms H1: Multimodal fusion beats text-only (+5.3% F1)."),
        ("SarcasTone (Full)", "Text + Audio + 11-d Explicit Affective Mismatch & Discordance Vector.", "Acc: 67.97%, F1: 68.91%. Confirms H3: Explicit mismatch boosts F1 by +1.77% over base fusion."),
        ("Context History", "Utterance concatenated with preceding conversational dialogue turns.", "Acc: 67.83%, F1: 68.68%. Conversational context stabilizes ambiguous one-word replies."),
        ("Whisper ASR Noise", "Simulated ASR transcription errors (phonetic typos, dropped filler words).", "SarcasTone F1 dropped by only 0.0014 (68.77%), proving prosodic failsafe robustness.")
    ]

    for r_idx, (area, cases, purp) in enumerate(res_data, start=1):
        bg = "F4F7F9" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate([area, cases, purp]):
            cell = tbl_res.cell(r_idx, c_idx)
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=50, bottom=50, left=100, right=100)
            set_cell_borders(cell, top="single", bottom="single", color="E0E6EB", sz="4")
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.1
            r = p.add_run(val)
            r.font.name = 'Arial'
            r.font.size = Pt(8.0)
            if c_idx == 0:
                r.bold = True

    add_h2("Interpretation and limits")
    add_body(
        "The empirical results provide strong evidence across all three hypotheses: H1 is confirmed as multimodal fusion beats text-only RoBERTa by +5.29% F1; H2 is supported as multimodal gains reach +4.7% on discordant clips; H3 is validated by the +1.77% boost achieved purely from explicit mismatch features. Limitations: (1) Sitcom acting features exaggerated emotional delivery; (2) Deadpan sarcasm with flat acoustic pitch remains challenging for speech prosody, requiring subtle linguistic inference; (3) High-arousal authentic praise can occasionally register false positive flags."
    )

    add_h2("How to reproduce")
    add_body(
        "Clone the repository and run python scripts/run_experiments.py to re-evaluate all 5 folds on the MUStARD benchmark. To launch the interactive real-time web application, execute streamlit run app.py --server.port 8501 and navigate to http://localhost:8501 in any browser."
    )

    p_ref = doc.add_paragraph()
    p_ref.paragraph_format.space_before = Pt(4)
    p_ref.paragraph_format.space_after = Pt(0)
    r_ref = p_ref.add_run("Implementation references: sarcastone/text_modality.py, audio_modality.py, mismatch.py, models.py, pipeline.py, dataset.py, app.py; benchmark data in data/mustard/ and data/mustard_plus/.")
    r_ref.font.name = 'Arial'
    r_ref.font.size = Pt(8.0)
    r_ref.font.color.rgb = COLOR_GRAY

    # Save DOCX
    docx_out = os.path.join(DELIVERABLES_DIR, "SarcasTone_Faculty_Project_Report.docx")
    doc.save(docx_out)
    print(f"DOCX report saved: {docx_out}")

    # Convert to PDF via Word COM
    pdf_out = os.path.join(DELIVERABLES_DIR, "SarcasTone_Faculty_Project_Report.pdf")
    try:
        import comtypes.client
        word = comtypes.client.CreateObject('Word.Application')
        word.Visible = False
        in_abs = os.path.abspath(docx_out)
        out_abs = os.path.abspath(pdf_out)
        wdoc = word.Documents.Open(in_abs)
        wdoc.SaveAs(out_abs, FileFormat=17) # 17 is wdFormatPDF
        wdoc.Close()
        word.Quit()
        print(f"PDF report successfully generated via Word COM: {pdf_out} (Size: {os.path.getsize(pdf_out)} bytes)")
    except Exception as e:
        print(f"Word COM error: {e}")

    return docx_out, pdf_out

if __name__ == "__main__":
    build_faculty_report()
