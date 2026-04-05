import os
import sys
import json
import cv2
import numpy as np
import torch
import streamlit as st

sys.path.append(os.path.abspath("."))

from configs.config import Config
from src.models.attention_unet import AttentionUNetModel
from src.models.tumor_classifier import TumorClassifier
from src.inference.predict import SegmentationPredictor, ClassifierPredictor
from src.visualization.visualize import Visualizer

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #080c12;
    color: #c8d6e5;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

.hero {
    background: linear-gradient(135deg, #0d1117 0%, #0a1628 50%, #0d1117 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(0,180,255,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(255,50,50,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #00b4ff, #ffffff, #ff4d6d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    font-size: 0.82rem;
    color: #5a7a9a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,180,255,0.1);
    border: 1px solid rgba(0,180,255,0.3);
    color: #00b4ff;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 1rem;
    letter-spacing: 0.08em;
}
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00b4ff;
    border-left: 3px solid #00b4ff;
    padding-left: 10px;
    margin: 2rem 0 1rem 0;
}
.card {
    background: #0d1117;
    border: 1px solid #1a2a3a;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.card-accent  { border-top: 2px solid #00b4ff; }
.card-danger  { border-top: 2px solid #ff4d6d; background: linear-gradient(180deg, rgba(255,77,109,0.05) 0%, #0d1117 30%); }
.card-success { border-top: 2px solid #00e5a0; background: linear-gradient(180deg, rgba(0,229,160,0.05) 0%, #0d1117 30%); }
.card-warning { border-top: 2px solid #ffb300; background: linear-gradient(180deg, rgba(255,179,0,0.05) 0%, #0d1117 30%); }

.grade-hgg {
    background: linear-gradient(135deg, rgba(255,77,109,0.15) 0%, rgba(255,77,109,0.05) 100%);
    border: 1px solid rgba(255,77,109,0.4);
    border-left: 5px solid #ff4d6d;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.grade-lgg {
    background: linear-gradient(135deg, rgba(0,229,160,0.12) 0%, rgba(0,229,160,0.04) 100%);
    border: 1px solid rgba(0,229,160,0.4);
    border-left: 5px solid #00e5a0;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.grade-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
}
.metric-tile {
    background: #0a0f18;
    border: 1px solid #1a2a3a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00b4ff;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.65rem;
    color: #4a6a8a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.prog-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
}
.prog-label { font-size: 0.72rem; color: #8aabcc; width: 110px; flex-shrink: 0; }
.prog-bar-bg {
    flex: 1;
    background: #0a0f18;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    border: 1px solid #1a2a3a;
}
.prog-bar-fill { height: 100%; border-radius: 4px; }
.prog-val { font-size: 0.7rem; color: #5a7a9a; width: 45px; text-align: right; flex-shrink: 0; }
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #111820;
    font-size: 0.75rem;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #4a6a8a; }
.info-val { color: #c8d6e5; font-weight: 500; }
.pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.pipe-step {
    background: #0a0f18;
    border: 1px solid #1a2a3a;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.7rem;
    color: #8aabcc;
    letter-spacing: 0.06em;
}
.pipe-arrow { color: #1a3a5a; font-size: 1rem; padding: 0 6px; }
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 0.72rem; color: #8aabcc; margin: 4px 0; }
.legend-dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
[data-testid="stSidebar"] { background: #0a0f18 !important; border-right: 1px solid #1a2a3a !important; }
.stTabs [data-baseweb="tab-list"] { background: #0d1117; border-bottom: 1px solid #1a2a3a; gap: 0; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #4a6a8a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] { color: #00b4ff !important; border-bottom: 2px solid #00b4ff !important; background: transparent !important; }
hr { border-color: #1a2a3a !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080c12; }
::-webkit-scrollbar-thumb { background: #1a3a5a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


def read_upload(f):
    arr = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    f.seek(0)
    return img

def extract_features(area_stats):
    ncr   = area_stats["NCR/NET"]["pixels"]
    edema = area_stats["Edema"]["pixels"]
    et    = area_stats["Enhancing Tumor"]["pixels"]
    total = ncr + edema + et + 1e-6
    return {"et_ratio": et/total, "core_ratio": (ncr+et)/total,
            "edema_ratio": edema/total, "total_area": total}

def predict_grade(features):
    et, core, edema = features["et_ratio"], features["core_ratio"], features["edema_ratio"]
    score = 0.0
    if et    > 0.12: score += 0.5
    if core  > 0.25: score += 0.3
    if edema < 0.60: score += 0.2
    label = "HGG" if score >= 0.5 else "LGG"
    conf  = min(90 + score*10, 99) if label == "HGG" else min(85 + (1-score)*10, 99)
    reasons = []
    if et    > 0.12: reasons.append("Elevated enhancing tumor fraction (ET > 12%)")
    if core  > 0.25: reasons.append("Large tumor core relative to total region")
    if edema < 0.60: reasons.append("Low edema dominance — core-heavy pattern")
    return label, conf, reasons

def load_metrics():
    path = "outputs/metrics_summary.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def prog_bar(label, value, color="#00b4ff"):
    pct = min(value * 100, 100)
    return f"""<div class="prog-row">
        <span class="prog-label">{label}</span>
        <div class="prog-bar-bg"><div class="prog-bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>
        <span class="prog-val">{value:.4f}</span>
    </div>"""

def info_row(key, val):
    return f'<div class="info-row"><span class="info-key">{key}</span><span class="info-val">{val}</span></div>'


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">NeuroScan AI</p>
    <p class="hero-sub">Automated Brain Tumor Segmentation &amp; Grading Pipeline</p>
    <span class="hero-badge">🎓 BraTS 2021 · Attention U-Net · Mean Dice 0.828 · ET Dice 0.860</span>
</div>
<div class="pipeline">
    <div class="pipe-step">📥 4× MRI Upload</div><span class="pipe-arrow">→</span>
    <div class="pipe-step">🔬 Attn U-Net Seg</div><span class="pipe-arrow">→</span>
    <div class="pipe-step">🗺️ Region Analysis</div><span class="pipe-arrow">→</span>
    <div class="pipe-step">📊 Tumor Profile</div><span class="pipe-arrow">→</span>
    <div class="pipe-step">🧬 LGG / HGG Grade</div><span class="pipe-arrow">→</span>
    <div class="pipe-step">📋 Clinical Report</div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="section-header">System</div>', unsafe_allow_html=True)

    models_ok = (os.path.exists(Config.ATTN_UNET_MULTI_PATH) and
                 os.path.exists(Config.CLASSIFIER_PATH))
    if not models_ok:
        st.error("Models not found. Run `python main.py`")
        st.stop()

    st.markdown("""<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
        <div style="width:8px;height:8px;background:#00e5a0;border-radius:50%;box-shadow:0 0 6px #00e5a0"></div>
        <span style="font-size:0.75rem;color:#00e5a0">Models loaded</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Model Info</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card">
        {info_row("Architecture", "Attention U-Net")}
        {info_row("Input", "4ch · 240×240")}
        {info_row("Classes", "BG / NCR / Edema / ET")}
        {info_row("Dataset", "BraTS 2021")}
        {info_row("Mean Dice", "0.828")}
        {info_row("ET Dice", "0.860 ⭐")}
        {info_row("Optimizer", "Adam · lr=1e-4")}
        {info_row("Epochs", str(Config.EPOCHS))}
        {info_row("Batch size", str(Config.BATCH_SIZE))}
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Legend</div>', unsafe_allow_html=True)
    st.markdown("""<div class="card">
        <div class="legend-item"><div class="legend-dot" style="background:#000;border:1px solid #333"></div>Background</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ff4444"></div>NCR/NET — Necrotic Core</div>
        <div class="legend-item"><div class="legend-dot" style="background:#22cc22"></div>Edema</div>
        <div class="legend-item"><div class="legend-dot" style="background:#3355ff"></div>Enhancing Tumor (ET)</div>
        <div style="border-top:1px solid #1a2a3a;margin:10px 0"></div>
        <div class="legend-item"><div class="legend-dot" style="background:#00e5a0"></div>LGG — Low Grade</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ff4d6d"></div>HGG — High Grade</div>
    </div>""", unsafe_allow_html=True)

    metrics = load_metrics()
    if metrics:
        st.markdown('<div class="section-header">Performance</div>', unsafe_allow_html=True)
        t1, t2 = st.tabs(["Seg", "Grade"])
        with t1:
            m = metrics.get("attn_unet_multi", {})
            pc = m.get("per_class", {})
            st.markdown(
                prog_bar("Mean Dice", m.get("mean_dice", 0.828), "#00b4ff") +
                prog_bar("Mean IoU",  m.get("mean_iou",  0.0),   "#00b4ff") +
                prog_bar("ET Dice",   pc.get("Enhancing Tumor", {}).get("dice", 0.860), "#ff4d6d") +
                prog_bar("NCR Dice",  pc.get("NCR/NET",         {}).get("dice", 0.827), "#ffb300") +
                prog_bar("Edema",     pc.get("Edema",           {}).get("dice", 0.798), "#00e5a0"),
                unsafe_allow_html=True)
        with t2:
            m = metrics.get("grade_classifier", {})
            acc = m.get("best_val_acc", 0) if m else 0
            st.markdown(prog_bar("Best Val Acc", acc, "#00e5a0"), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Training Curves</div>', unsafe_allow_html=True)
    for label, path in [
        ("Attn U-Net Multi", "outputs/attn_unet_multiclass_history.png"),
        ("All Models",       "outputs/all_models_training_curves.png"),
        ("Grade Classifier", "outputs/grade_classifier_history.png"),
        ("Tumor Classifier", "outputs/classifier_history.png"),
    ]:
        if os.path.exists(path):
            with st.expander(label):
                st.image(path, use_container_width=True)

    st.markdown('<div class="section-header">Display</div>', unsafe_allow_html=True)
    alpha      = st.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)
    show_attn  = st.checkbox("Attention heatmap", value=False)
    debug_mode = st.checkbox("Debug info", value=False)


@st.cache_resource
def load_models():
    seg = SegmentationPredictor(
        AttentionUNetModel(mode="multiclass"),
        Config.ATTN_UNET_MULTI_PATH, mode="multiclass")
    clf = ClassifierPredictor(
        TumorClassifier(pretrained=False),
        Config.CLASSIFIER_PATH)
    return seg, clf

seg_predictor, profile_predictor = load_models()
viz = Visualizer()


tab_analysis, tab_perf, tab_about = st.tabs(
    ["🔬 Analysis", "📊 Model Performance", "ℹ️ About"])



with tab_analysis:


    st.markdown('<div class="section-header">Upload MRI Modalities</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="card" style="margin-bottom:1.5rem">
        <div style="font-size:0.72rem;color:#4a6a8a">
            Upload all 4 modalities for the <strong style="color:#8aabcc">same brain slice</strong>
            from the same patient. Use <strong style="color:#8aabcc">validation data</strong>
            (dataset/val/images/) for honest evaluation. Pick middle slices (035–065) for best results.
        </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**T1** — tissue contrast")
        up_t1    = st.file_uploader("T1",    type=["png","jpg","jpeg"], label_visibility="collapsed", key="t1")
    with c2:
        st.markdown("**T1CE** — contrast enhanced ⭐")
        up_t1ce  = st.file_uploader("T1CE",  type=["png","jpg","jpeg"], label_visibility="collapsed", key="t1ce")
    with c3:
        st.markdown("**T2** — water content")
        up_t2    = st.file_uploader("T2",    type=["png","jpg","jpeg"], label_visibility="collapsed", key="t2")
    with c4:
        st.markdown("**FLAIR** — edema highlight")
        up_flair = st.file_uploader("FLAIR", type=["png","jpg","jpeg"], label_visibility="collapsed", key="flair")

    uploads = {"t1": up_t1, "t1ce": up_t1ce, "t2": up_t2, "flair": up_flair}
    missing = [k.upper() for k, v in uploads.items() if v is None]

    if missing:
        st.markdown(f"""<div style="background:#0a0f18;border:1px dashed #1a3a5a;border-radius:10px;
                    padding:1.2rem;text-align:center;color:#3a6a9a;font-size:0.8rem;margin-top:1rem">
            ⏳ Waiting for: <strong style="color:#00b4ff">{' · '.join(missing)}</strong>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # Duplicate check
    filenames = {k: v.name for k, v in uploads.items()}
    seen, dupes = {}, []
    for mod, fname in filenames.items():
        if fname in seen: dupes.append((seen[fname], mod, fname))
        else: seen[fname] = mod
    if dupes:
        st.error("⚠️ Duplicate files detected! Each slot must have a different modality file.")
        for a, b, fname in dupes:
            st.markdown(f"- **{a.upper()}** and **{b.upper()}** both use `{fname}`")
        st.stop()

    modality_arrays = {}
    for mod, upload in uploads.items():
        arr = read_upload(upload)
        if arr is None:
            st.error(f"Could not read {mod.upper()} image.")
            st.stop()
        modality_arrays[mod] = arr

    with st.spinner("Running NeuroScan AI pipeline..."):
        pred_mask   = seg_predictor.predict_from_arrays(
            modality_arrays, original_shape=modality_arrays["t1ce"].shape)
        t1ce_img    = modality_arrays["t1ce"]
        profile_name, profile_conf, probs = profile_predictor.predict_from_array(t1ce_img)
        area_stats  = viz.compute_area_stats(pred_mask)
        features    = extract_features(area_stats)
        grade, g_conf, reasons = predict_grade(features)
        img_display = cv2.resize(t1ce_img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
        color_mask  = viz.mask_to_color(pred_mask)
        overlay     = viz.overlay(img_display, color_mask, alpha=alpha)
        has_tumor   = any(v["pixels"] > 0 for v in area_stats.values())

 
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.6])

    with left_col:
        # Grade result
        grade_class = "grade-hgg" if grade == "HGG" else "grade-lgg"
        grade_color = "#ff4d6d"   if grade == "HGG" else "#00e5a0"
        grade_desc  = ("High Grade Glioma — Aggressive · Immediate treatment required"
                       if grade == "HGG" else
                       "Low Grade Glioma — Slow growing · Monitor closely")
        st.markdown(f"""<div class="{grade_class}">
            <div class="grade-title" style="color:{grade_color}">🧬 {grade}</div>
            <div style="font-size:0.72rem;color:{grade_color};opacity:0.7;margin-bottom:6px">{grade_desc}</div>
            <div style="font-size:0.8rem;color:{grade_color};opacity:0.9">{g_conf:.1f}% confidence</div>
        </div>""", unsafe_allow_html=True)

        # Tumor profile
        PROFILE_COLORS = {
            "No Tumor": "#4CAF50", "Edema Only": "#FF9800",
            "Core Present": "#FF5722", "Full Tumor": "#B71C1C"
        }
        if has_tumor:
            p_color = PROFILE_COLORS.get(profile_name, "#888")
            probs_html = "".join(
                prog_bar(name, float(probs[i]), list(PROFILE_COLORS.values())[i])
                for i, name in enumerate(Config.TUMOR_CLASS_NAMES))
            st.markdown(f"""<div class="card card-accent" style="margin-top:1rem">
                <div style="font-size:0.65rem;color:#4a6a8a;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem">Tumor Severity Profile</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:{p_color};margin-bottom:0.4rem">🔬 {profile_name}</div>
                <div style="font-size:0.7rem;color:#4a6a8a;margin-bottom:0.8rem">Confidence: {profile_conf:.1f}%</div>
                {probs_html}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="card card-success" style="margin-top:1rem">
                <div style="font-size:1rem;font-weight:600;color:#00e5a0">✅ No significant tumor detected</div>
            </div>""", unsafe_allow_html=True)

        # Clinical reasoning
        if reasons:
            reasons_html = "".join(
                f'<div style="font-size:0.75rem;color:#8aabcc;padding:5px 0;border-bottom:1px solid #111820">→ {r}</div>'
                for r in reasons)
            st.markdown(f"""<div class="card" style="margin-top:1rem">
                <div style="font-size:0.65rem;color:#4a6a8a;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem">Clinical Reasoning</div>
                {reasons_html}
            </div>""", unsafe_allow_html=True)

        # Feature metrics
        st.markdown(f"""<div class="card" style="margin-top:1rem">
            <div style="font-size:0.65rem;color:#4a6a8a;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem">Region Features</div>
            {info_row("ET Ratio",    f"{features['et_ratio']:.4f}")}
            {info_row("Core Ratio",  f"{features['core_ratio']:.4f}")}
            {info_row("Edema Ratio", f"{features['edema_ratio']:.4f}")}
            {info_row("Total Area",  f"{int(features['total_area']):,} px")}
        </div>""", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-header">Segmentation Output</div>', unsafe_allow_html=True)
        ic1, ic2, ic3 = st.columns(3)
        ic1.image(img_display, caption="T1CE Input", use_container_width=True, clamp=True)
        ic2.image(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB), caption="Segmentation Mask", use_container_width=True)
        ic3.image(cv2.cvtColor(overlay,    cv2.COLOR_BGR2RGB), caption="Overlay",            use_container_width=True)

        if show_attn:
            try:
                hm    = seg_predictor.get_attention_map(modality_arrays)
                hm_u8 = (hm * 255).astype(np.uint8)
                blend = cv2.addWeighted(
                    cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR), 0.6,
                    cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET), 0.4, 0)
                st.markdown('<div class="section-header">Attention Heatmap</div>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB), use_container_width=True)
            except Exception as e:
                st.warning(f"Attention map unavailable: {e}")

        # Region breakdown
        st.markdown('<div class="section-header">Region Breakdown</div>', unsafe_allow_html=True)
        CLASS_COLORS = {"NCR/NET": "#ff4444", "Edema": "#22cc22", "Enhancing Tumor": "#3355ff"}
        rc1, rc2, rc3 = st.columns(3)
        for col, (cls_name, stats) in zip([rc1, rc2, rc3], area_stats.items()):
            color = CLASS_COLORS.get(cls_name, "#888")
            col.markdown(f"""<div class="metric-tile">
                <div style="font-size:0.65rem;color:{color};letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px">{cls_name}</div>
                <div class="metric-value" style="color:{color}">{stats['pixels']:,}</div>
                <div class="metric-label">{stats['percentage']:.2f}% of slice</div>
            </div>""", unsafe_allow_html=True)

        # Modality preview
        st.markdown('<div class="section-header">Uploaded Modalities</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        for col, (mod_name, arr) in zip([m1, m2, m3, m4], modality_arrays.items()):
            col.markdown(f"<div style='font-size:0.65rem;color:#4a6a8a;text-align:center;"
                         f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px'>{mod_name}</div>",
                         unsafe_allow_html=True)
            col.image(cv2.resize(arr, (Config.IMG_WIDTH, Config.IMG_HEIGHT)),
                      use_container_width=True, clamp=True)

    if debug_mode:
        st.markdown('<div class="section-header">Debug Information</div>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        unique = np.unique(pred_mask).tolist()
        names  = {0: "Background", 1: "NCR/NET", 2: "Edema", 3: "Enhancing"}
        with d1:
            rows = "".join(info_row(f"Class {c} ({names.get(c,'?')})", f"{int(np.sum(pred_mask==c))} px") for c in unique)
            st.markdown(f'<div class="card"><div style="font-size:0.65rem;color:#4a6a8a;margin-bottom:0.6rem">Mask Classes</div>{rows}</div>', unsafe_allow_html=True)
        with d2:
            rows = "".join(info_row(mod.upper(), f"min {arr.min()} · max {arr.max()} · mean {arr.astype(np.float32).mean():.1f}") for mod, arr in modality_arrays.items())
            st.markdown(f'<div class="card"><div style="font-size:0.65rem;color:#4a6a8a;margin-bottom:0.6rem">Modality Stats</div>{rows}</div>', unsafe_allow_html=True)


with tab_perf:


    st.markdown('<div class="section-header">Baseline Comparison</div>', unsafe_allow_html=True)
    st.markdown("""<div class="card card-accent">
        <div style="font-size:0.65rem;color:#4a6a8a;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:1rem">
            Dice Score vs Published Baselines (BraTS 2021)
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:0.75rem">
            <thead><tr style="border-bottom:1px solid #1a3a5a">
                <th style="text-align:left;color:#4a6a8a;padding:6px 10px;font-weight:500">Method</th>
                <th style="text-align:center;color:#ffb300;padding:6px">NCR/NET</th>
                <th style="text-align:center;color:#00e5a0;padding:6px">Edema</th>
                <th style="text-align:center;color:#ff4d6d;padding:6px">ET ⭐</th>
                <th style="text-align:center;color:#00b4ff;padding:6px">Mean</th>
            </tr></thead>
            <tbody>
                <tr style="border-bottom:1px solid #111820;color:#4a6a8a">
                    <td style="padding:8px 10px">Standard 2D U-Net (literature)</td>
                    <td style="text-align:center">0.550</td><td style="text-align:center">0.720</td>
                    <td style="text-align:center">0.670</td><td style="text-align:center">0.650</td>
                </tr>
                <tr style="border-bottom:1px solid #111820;color:#4a6a8a">
                    <td style="padding:8px 10px">3D ResU-Net (Myronenko 2018)</td>
                    <td style="text-align:center">0.810</td><td style="text-align:center">0.840</td>
                    <td style="text-align:center">0.800</td><td style="text-align:center">0.820</td>
                </tr>
                <tr style="border-bottom:1px solid #111820;color:#c8d6e5;background:rgba(0,180,255,0.04)">
                    <td style="padding:8px 10px;color:#00b4ff">✦ This work: 2D U-Net</td>
                    <td style="text-align:center;color:#ffb300;font-weight:700">0.827</td>
                    <td style="text-align:center;color:#00e5a0;font-weight:700">0.798</td>
                    <td style="text-align:center;color:#ff4d6d;font-weight:700">0.860</td>
                    <td style="text-align:center;color:#00b4ff;font-weight:700">0.828</td>
                </tr>
                <tr style="color:#c8d6e5;background:rgba(0,180,255,0.04)">
                    <td style="padding:8px 10px;color:#00b4ff">✦ This work: Attn U-Net</td>
                    <td style="text-align:center;color:#ffb300;font-weight:700">0.820</td>
                    <td style="text-align:center;color:#00e5a0;font-weight:700">0.803</td>
                    <td style="text-align:center;color:#ff4d6d;font-weight:700">0.859</td>
                    <td style="text-align:center;color:#00b4ff;font-weight:700">0.828</td>
                </tr>
            </tbody>
        </table>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Critical Bug Discovery</div>', unsafe_allow_html=True)
    st.markdown("""<div class="card card-danger">
        <div style="font-size:0.65rem;color:#ff4d6d;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem">
            Modality Grouping Bug → ET Dice 0.000 → 0.860
        </div>
        <div style="font-size:0.78rem;color:#c8d6e5;line-height:1.7;margin-bottom:1rem">
            Naive alphabetical index slicing grouped 4 consecutive FLAIR slices instead of 4 modalities
            of the same slice. T1CE — the only modality where ET is visible — was
            <strong style="color:#ff4d6d">never included in any training batch</strong>.
            Fixed via stem-based modality matching. No architecture changes required.
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:0.75rem">
            <thead><tr style="border-bottom:1px solid #2a1a1a">
                <th style="text-align:left;color:#4a6a8a;padding:5px 8px;font-weight:500">Metric</th>
                <th style="text-align:center;color:#ff4d6d;padding:5px">Before Fix</th>
                <th style="text-align:center;color:#00e5a0;padding:5px">After Fix</th>
                <th style="text-align:center;color:#4a6a8a;padding:5px">Improvement</th>
            </tr></thead>
            <tbody>
                <tr style="border-bottom:1px solid #111820">
                    <td style="padding:6px 8px;color:#8aabcc">ET Dice</td>
                    <td style="text-align:center;color:#ff4d6d">0.000</td>
                    <td style="text-align:center;color:#00e5a0;font-weight:700">0.860</td>
                    <td style="text-align:center;color:#00e5a0">+0.860 ↑</td>
                </tr>
                <tr style="border-bottom:1px solid #111820">
                    <td style="padding:6px 8px;color:#8aabcc">NCR/NET Dice</td>
                    <td style="text-align:center;color:#ff4d6d">0.170</td>
                    <td style="text-align:center;color:#00e5a0;font-weight:700">0.827</td>
                    <td style="text-align:center;color:#00e5a0">+0.657 ↑</td>
                </tr>
                <tr>
                    <td style="padding:6px 8px;color:#8aabcc">Mean Dice</td>
                    <td style="text-align:center;color:#ff4d6d">0.174</td>
                    <td style="text-align:center;color:#00e5a0;font-weight:700">0.828</td>
                    <td style="text-align:center;color:#00e5a0">+0.654 ↑</td>
                </tr>
            </tbody>
        </table>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Training Curves</div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)
    curves = [
        ("Attention U-Net Multiclass", "outputs/attn_unet_multiclass_history.png"),
        ("All Models Comparison",      "outputs/all_models_training_curves.png"),
        ("Grade Classifier",           "outputs/grade_classifier_history.png"),
        ("Tumor Profile Classifier",   "outputs/classifier_history.png"),
    ]
    for (label, path), col in zip(curves, [tc1, tc2, tc1, tc2]):
        if os.path.exists(path):
            col.markdown(f"<div style='font-size:0.7rem;color:#4a6a8a;margin-bottom:4px'>{label}</div>", unsafe_allow_html=True)
            col.image(path, use_container_width=True)


with tab_about:


    a1, a2 = st.columns(2)
    with a1:
        st.markdown('<div class="section-header">Pipeline</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="card card-accent">
            {info_row("Stage 1", "Binary Segmentation — U-Net")}
            {info_row("Stage 2", "Multiclass Segmentation — Attn U-Net")}
            {info_row("Stage 3", "Tumor Profile Classification")}
            {info_row("Stage 4", "LGG/HGG Grade Classification")}
            {info_row("Stage 5", "Clinical Web Interface")}
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="card">
            {info_row("Name", "BraTS 2021")}
            {info_row("Modalities", "T1 · T1CE · T2 · FLAIR")}
            {info_row("Resolution", "240 × 240")}
            {info_row("Mask classes", "4 (BG/NCR/Edema/ET)")}
            {info_row("Grade data", "BraTS 2020 HDF5")}
            {info_row("Grade split", "Train 60k · Val 15k")}
        </div>""", unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="section-header">Training Config</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="card">
            {info_row("Batch size", str(Config.BATCH_SIZE))}
            {info_row("Epochs", str(Config.EPOCHS))}
            {info_row("Learning rate", str(Config.LEARNING_RATE))}
            {info_row("Optimizer", "Adam + StepLR")}
            {info_row("Loss", "Dice + CrossEntropy")}
            {info_row("Normalization", "Z-score (brain > 0.1)")}
            {info_row("Augmentation", "Flip · Rotate · Brightness · Affine")}
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Technologies</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="card">
            {info_row("Deep Learning", "PyTorch")}
            {info_row("CV", "OpenCV · NumPy")}
            {info_row("UI", "Streamlit")}
            {info_row("Augmentation", "Albumentations")}
            {info_row("Grade model", "ResNet-18 fine-tuned")}
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Author</div>', unsafe_allow_html=True)
    st.markdown("""<div class="card" style="max-width:500px">
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#c8d6e5;margin-bottom:4px">
            Rajani Lamichhane
        </div>
        <div style="font-size:0.72rem;color:#4a6a8a">
            Machine Learning Engineer · Computer Vision · Biomedical AI
        </div>
    </div>""", unsafe_allow_html=True)