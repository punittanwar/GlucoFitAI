import streamlit as st
import joblib
import pandas as pd
import html as html_mod
from youtube_helper import fetch_videos, get_personalized_category, CATEGORY_QUERIES

# ── Pre-warm model, scaler & data (cached across all pages) ──────────────────
@st.cache_resource
def _load_model(): return joblib.load('diabetes_model.pkl')
@st.cache_resource
def _load_scaler(): return joblib.load('scaler.pkl')
@st.cache_data
def _load_diabetes_data():
    df = pd.read_csv('diabetes.csv')
    def _assign_diet(row):
        g, b, i = row['Glucose'], row['BMI'], row['Insulin']
        if g > 140: return 0
        if b < 25 and g <= 110: return 1
        if b >= 33 and g <= 140: return 2
        return 3
    df['DietType'] = df.apply(_assign_diet, axis=1)
    return df

# Trigger caching NOW so Assessment page loads instantly
_load_model()
_load_scaler()
_load_diabetes_data()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlucoFit AI – Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.html("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .block-container {
    background: #030d1a !important;
    color: #e8f4fd !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"]          { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Ambient background blobs ── */
body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 60% 50% at 15% 20%, rgba(0,200,170,.13) 0%, transparent 70%),
        radial-gradient(ellipse 50% 45% at 85% 75%, rgba(0,120,255,.12) 0%, transparent 70%),
        radial-gradient(ellipse 40% 35% at 50% 50%, rgba(0,220,200,.06) 0%, transparent 70%);
}

/* ── Section wrapper ── */
.section { position: relative; z-index: 1; }

/* ══════════════════════════════════════════════════
   HERO
══════════════════════════════════════════════════ */
.hero {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center;
    padding: 90px 20px 60px;
    position: relative; z-index: 1;
}

.badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(0,200,170,.12);
    border: 1px solid rgba(0,200,170,.35);
    border-radius: 100px;
    padding: 6px 18px;
    font-family: 'Sora', sans-serif;
    font-size: .88rem; font-weight: 600;
    letter-spacing: .08em; text-transform: uppercase;
    color: #00c8aa;
    margin-bottom: 28px;
    animation: fadeSlideDown .6s ease both;
}

.brand-title {
    display: block;
    font-family: 'Sora', sans-serif;
    font-size: clamp(3.2rem, 8vw, 6.5rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #00e5c4 0%, #0091ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 60px rgba(0, 145, 255, 0.4);
    margin-bottom: 24px;
    animation: fadeSlideDown .7s ease both;
}

.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: clamp(2.2rem, 5vw, 4rem);
    font-weight: 700;
    line-height: 1.08;
    letter-spacing: -.02em;
    color: #ffffff;
    margin-bottom: 12px;
    animation: fadeSlideDown .7s .1s ease both;
}

.hero-title span {
    background: linear-gradient(135deg, #00e5c4 0%, #00a8ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: clamp(1.1rem, 2.2vw, 1.35rem);
    font-weight: 300;
    color: #8db8d8;
    max-width: 620px;
    line-height: 1.7;
    margin-bottom: 44px;
    animation: fadeSlideDown .7s .2s ease both;
}

/* CTA Buttons */
.cta-row {
    display: flex; gap: 16px; flex-wrap: wrap;
    justify-content: center;
    animation: fadeSlideDown .7s .3s ease both;
}

.btn-primary {
    display: inline-flex; align-items: center; gap: 8px;
    background: linear-gradient(135deg, #00c8aa, #0091ff);
    color: #fff; font-family: 'Sora', sans-serif;
    font-weight: 600; font-size: .95rem;
    padding: 14px 32px; border-radius: 12px;
    text-decoration: none;
    box-shadow: 0 4px 24px rgba(0,200,170,.35);
    transition: transform .2s, box-shadow .2s;
}
.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,200,170,.5);
    color: #fff; text-decoration: none;
}

.btn-secondary {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.15);
    color: #c8dff0; font-family: 'Sora', sans-serif;
    font-weight: 500; font-size: .95rem;
    padding: 14px 32px; border-radius: 12px;
    text-decoration: none;
    transition: background .2s, transform .2s;
}
.btn-secondary:hover {
    background: rgba(255,255,255,.1);
    transform: translateY(-2px);
    color: #fff; text-decoration: none;
}

/* ── Stat strip ── */
.stats-strip {
    display: flex; justify-content: center; gap: 32px; flex-wrap: wrap;
    padding: 32px 20px;
    border-top: 1px solid rgba(255,255,255,.06);
    border-bottom: 1px solid rgba(255,255,255,.06);
    margin: 0 4%;
    animation: fadeIn .8s .5s ease both;
}
.stat { text-align: center; }
.stat-num {
    font-family: 'Sora', sans-serif;
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #00e5c4, #00a8ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label { font-size: .95rem; color: #5a7f9a; font-weight: 400; margin-top: 4px; }


/* ══════════════════════════════════════════════════
   FEATURES CARDS
══════════════════════════════════════════════════ */
.features { padding: 20px 5% 80px; }

.feat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 20px;
}

.feat-card {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 20px;
    padding: 32px 28px;
    transition: transform .25s, border-color .25s, background .25s;
    position: relative; overflow: hidden;
}
.feat-card::before {
    content: '';
    position: absolute; inset: 0; border-radius: 20px;
    opacity: 0; transition: opacity .3s;
    background: radial-gradient(ellipse at top left, rgba(0,200,170,.08), transparent 60%);
}
.feat-card:hover { transform: translateY(-4px); border-color: rgba(0,200,170,.25); }
.feat-card:hover::before { opacity: 1; }

.feat-icon { font-size: 1.8rem; margin-bottom: 16px; }
.feat-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.08rem; font-weight: 700; color: #d4eeff;
    margin-bottom: 10px;
}
.feat-text { font-size: .97rem; color: #5a7f9a; line-height: 1.7; }

/* ══════════════════════════════════════════════════
   VISUAL SECTION (images)
══════════════════════════════════════════════════ */
.visual-section { padding: 20px 5% 80px; }

.visual-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
}

.visual-card {
    border-radius: 16px; overflow: hidden;
    border: 1px solid rgba(255,255,255,.07);
    background: rgba(255,255,255,.02);
    position: relative;
}
.visual-card img {
    width: 100%; height: 160px;
    object-fit: cover; display: block;
    filter: brightness(.85) saturate(1.1);
    transition: transform .4s, filter .4s;
}
.visual-card:hover img { transform: scale(1.04); filter: brightness(.95) saturate(1.2); }
.visual-caption {
    padding: 16px 20px;
    font-size: .95rem; color: #5a7f9a;
    font-style: italic;
}

/* ══════════════════════════════════════════════════
   VIDEO SECTION
══════════════════════════════════════════════════ */
.video-section { padding: 0 5% 80px; }

[data-testid="stVideo"] {
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(0,200,170,.2);
    box-shadow: 0 0 60px rgba(0,200,170,.1);
}

/* ══════════════════════════════════════════════════
   ANIMATIONS
══════════════════════════════════════════════════ */
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

/* Section toggle button row */
.section-btn-row {
    display: flex; gap: 14px; flex-wrap: wrap;
    justify-content: center;
    padding: 44px 5% 8px;
}

/* ── Compact toggle-pill buttons ── */
[data-testid="stHorizontalBlock"]:has([data-testid="stBaseButton-secondary"]) .stButton > button,
.toggle-btn-area .stButton > button {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid rgba(0,200,170,.3) !important;
    color: #a0d8ef !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 500 !important;
    font-size: .85rem !important;
    padding: 10px 22px !important;
    border-radius: 100px !important;
    box-shadow: none !important;
    transition: background .25s, border-color .25s, transform .2s, color .2s !important;
    letter-spacing: .02em !important;
}
[data-testid="stHorizontalBlock"]:has([data-testid="stBaseButton-secondary"]) .stButton > button:hover,
.toggle-btn-area .stButton > button:hover {
    background: rgba(0,200,170,.12) !important;
    border-color: rgba(0,200,170,.55) !important;
    color: #00e5c4 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,200,170,.15) !important;
}

/* ══════════════════════════════════════════════════
   RESPONSIVE — Tablet ≤ 960px
══════════════════════════════════════════════════ */
@media (max-width: 960px) {
    .feat-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .visual-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .stats-strip {
        gap: 24px;
    }
    /* Streamlit 3-col video rows → wrap to 1 column on tablet */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: min(280px, 100%) !important;
    }
}

/* ══════════════════════════════════════════════════
   RESPONSIVE — Mobile ≤ 640px
══════════════════════════════════════════════════ */
@media (max-width: 640px) {

    /* Hero */
    .hero { padding: 60px 16px 40px; }
    .hero-sub { font-size: 1rem; margin-bottom: 28px; }
    .badge { font-size: .78rem; padding: 5px 14px; }

    /* Stats */
    .stats-strip {
        gap: 18px 32px;
        padding: 24px 16px;
        margin: 0 2%;
    }
    .stat-num { font-size: 1.6rem; }
    .stat-label { font-size: .85rem; }

    /* Feature cards — single column */
    .feat-grid {
        grid-template-columns: 1fr;
    }
    .feat-card { padding: 24px 20px; }
    .features { padding: 16px 4% 40px; }

    /* Visual images — single column */
    .visual-grid {
        grid-template-columns: 1fr;
    }
    .visual-card img { height: 200px; }
    .visual-section { padding: 16px 4% 40px; }
    .visual-caption { font-size: .9rem; }

    /* Video section */
    .video-section { padding: 0 4% 40px; }
    [data-testid="stVideo"] { border-radius: 12px; }

    /* Streamlit columns — stack vertically on mobile */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 12px !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* Tabs — scrollable on mobile */
    [data-testid="stTabs"] [role="tablist"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
    }
    [data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar { display: none; }
    [data-testid="stTabs"] [role="tab"] {
        white-space: nowrap;
        font-size: .82rem !important;
        padding: 8px 10px !important;
    }

    /* Streamlit button — full width */
    .stButton > button {
        width: 100% !important;
        font-size: .9rem !important;
        padding: 12px 16px !important;
    }

    /* Yoga / video search bar */
    [data-testid="stTextInput"] input[type="text"] {
        font-size: .9rem !important;
    }

    /* Yoga card channel text */
    .yv-channel { font-size: .8rem; }
    .yv-title { font-size: .92rem; }

    /* Stat strip — 2 across on small phones */
    .stat { flex: 1 1 calc(50% - 16px); }
}

/* ── Streamlit button override ── */
.stButton > button {
    background: linear-gradient(135deg, #00c8aa, #0091ff) !important;
    color: #fff !important; border: none !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important; font-size: .95rem !important;
    padding: 14px 36px !important; border-radius: 12px !important;
    box-shadow: 0 4px 24px rgba(0,200,170,.3) !important;
    transition: transform .2s, box-shadow .2s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,200,170,.5) !important;
}

/* ══════════════════════════════════════════════════
   YOGA SECTION — Dynamic Video Library
══════════════════════════════════════════════════ */

/* Category header banner */
.ycat-header {
    display: flex; align-items: center; gap: 16px;
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
    position: relative; overflow: hidden;
}
.ycat-accent {
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.ycat-accent-fire   { background: linear-gradient(90deg, #f97316, #fb923c); }
.ycat-accent-cyan   { background: linear-gradient(90deg, #06b6d4, #3b82f6); }
.ycat-accent-purple { background: linear-gradient(90deg, #a855f7, #ec4899); }
.ycat-accent-green  { background: linear-gradient(90deg, #10b981, #34d399); }

.ycat-icon { font-size: 1.8rem; flex-shrink: 0; }
.ycat-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.05rem; font-weight: 700;
    color: #d4eeff; margin: 0;
}

/* ── Video Card (thumbnail-based) ── */
.yv-card {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 16px;
    overflow: hidden;
    transition: transform .25s, border-color .25s, box-shadow .25s;
    cursor: pointer;
}
.yv-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0,200,170,.3);
    box-shadow: 0 8px 32px rgba(0,200,170,.12);
}
.yv-thumb-wrap {
    position: relative; overflow: hidden;
    aspect-ratio: 16/9;
}
.yv-thumb-wrap img {
    width: 100%; height: 100%;
    object-fit: cover; display: block;
    transition: transform .4s;
}
.yv-card:hover .yv-thumb-wrap img {
    transform: scale(1.05);
}
.yv-play-btn {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 56px; height: 56px;
    background: rgba(0,0,0,.55);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    opacity: 0; transition: opacity .25s;
    backdrop-filter: blur(4px);
}
.yv-card:hover .yv-play-btn { opacity: 1; }
.yv-play-btn::after {
    content: '';
    display: block;
    width: 0; height: 0;
    border-top: 10px solid transparent;
    border-bottom: 10px solid transparent;
    border-left: 18px solid #fff;
    margin-left: 4px;
}

.yv-info { padding: 14px 16px 16px; }
.yv-title {
    font-family: 'Sora', sans-serif;
    font-size: .97rem; font-weight: 600;
    color: #c8e6ff;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: 6px;
}
.yv-channel {
    font-size: .86rem; color: #4a7a9a;
    display: flex; align-items: center; gap: 4px;
}
.yv-channel::before {
    content: '●'; font-size: .5rem; color: #00c8aa;
}

/* ── Search bar override ── */
[data-testid="stTextInput"] input[type="text"] {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 12px !important;
    color: #c8e6ff !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 12px 16px !important;
}
[data-testid="stTextInput"] input[type="text"]:focus {
    border-color: rgba(0,200,170,.4) !important;
    box-shadow: 0 0 0 2px rgba(0,200,170,.15) !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #3a6080 !important;
}

/* Video caption card below embedded player */
.vcap {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 0 0 14px 14px;
    padding: 14px 16px 16px;
    margin-top: -8px;
}
.vcap-name {
    font-family: 'Sora', sans-serif;
    font-size: .92rem; font-weight: 700;
    color: #c8e6ff;
    margin-bottom: 6px;
}
.vcap-sk {
    font-size: .86rem; font-weight: 400;
    color: #3a6080; font-style: italic;
    margin-left: 6px;
}
.vcap-tag {
    display: inline-block;
    font-size: .78rem; font-weight: 700;
    letter-spacing: .06em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 100px;
    margin-bottom: 8px;
    white-space: nowrap;
}
.vtag-fire   { background: rgba(249,115,22,.12); border: 1px solid rgba(249,115,22,.3); color: #fb923c; }
.vtag-cyan   { background: rgba(6,182,212,.12);  border: 1px solid rgba(6,182,212,.3);  color: #22d3ee; }
.vtag-purple { background: rgba(168,85,247,.12); border: 1px solid rgba(168,85,247,.3); color: #c084fc; }
.vtag-green  { background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.3); color: #34d399; }

.vcap-ben {
    font-size: .92rem; color: #5a7f9a;
    line-height: 1.55;
}

/* ── Loading pulse ── */
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.yv-loading {
    background: linear-gradient(90deg, rgba(255,255,255,.03) 25%, rgba(255,255,255,.06) 50%, rgba(255,255,255,.03) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s ease infinite;
    border-radius: 16px;
    aspect-ratio: 16/9;
    margin-bottom: 12px;
}

/* ── No results ── */
.yv-empty {
    text-align: center; padding: 60px 24px;
    color: #4a7a9a;
    font-family: 'Sora', sans-serif;
    font-size: .95rem;
}
.yv-empty-icon { font-size: 2.5rem; margin-bottom: 12px; display: block; }
</style>
""")


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.html("""
<div class="section hero">

  <div class="brand-title">GlucoFit AI</div>

  <div class="badge">🩺 &nbsp; AI-Powered Health Intelligence</div>
  <h1 class="hero-title">
    Know Your Risk.<br>
    <span>Take Control Early.</span>
  </h1>

  <p class="hero-sub">
    GlucoFit AI blends clinical biomarkers, lifestyle factors, and advanced
    machine learning to deliver a personalised diabetes risk assessment —
    plus an AI-generated report you can share with your doctor.
  </p>

</div>
""")

# ── Hero CTA buttons (Streamlit native for working navigation) ────────────────
st.html("<div style='height:40px'></div>")
_hero_l, _hero_btn1, _hero_gap, _hero_btn2, _hero_r = st.columns([3, 2, 0.5, 2, 3])
with _hero_btn1:
    if st.button("🚀  Start Assessment", use_container_width=True, key='hero_start'):
        st.switch_page("pages/2_Assessment.py")
with _hero_btn2:
    if st.button("📖  How It Works", use_container_width=True, key='hero_how_it_works'):
        st.switch_page("pages/1_How_It_Works.py")


# ── Stat strip ────────────────────────────────────────────────────────────────
st.html("""
<div class="stats-strip">
  <div class="stat"><div class="stat-num">537M+</div><div class="stat-label">People living with diabetes globally</div></div>
  <div class="stat"><div class="stat-num">50%</div><div class="stat-label">Cases undiagnosed</div></div>
  <div class="stat"><div class="stat-num">80%</div><div class="stat-label">Type-2 cases preventable</div></div>
  <div class="stat"><div class="stat-num">95%</div><div class="stat-label">Model accuracy on test data</div></div>
</div>
""")


# ══════════════════════════════════════════════════════════════════════════════
# 3 SECTION TOGGLE BUTTONS
# ══════════════════════════════════════════════════════════════════════════════
for _k in ['show_why', 'show_understand', 'show_learn']:
    if _k not in st.session_state:
        st.session_state[_k] = False

st.html("<div style='height:40px'></div>")
_pad_l, _sb1, _sb2, _sb3, _pad_r = st.columns([2, 1.5, 1.5, 1.5, 2])
with _sb1:
    _why_label = "💡 Hide: Why GlucoFit AI?" if st.session_state['show_why'] else "💡 Why GlucoFit AI?"
    if st.button(_why_label, key='btn_why', use_container_width=True):
        _new = not st.session_state['show_why']
        st.session_state['show_why'] = _new
        # Accordion: close the other sections when opening this one
        if _new:
            st.session_state['show_understand'] = False
            st.session_state['show_learn'] = False
        st.rerun()
with _sb2:
    _und_label = "🩺 Hide: Understanding Diabetes" if st.session_state['show_understand'] else "🩺 Understanding Diabetes"
    if st.button(_und_label, key='btn_understand', use_container_width=True):
        _new = not st.session_state['show_understand']
        st.session_state['show_understand'] = _new
        # Accordion: close the other sections when opening this one
        if _new:
            st.session_state['show_why'] = False
            st.session_state['show_learn'] = False
        st.rerun()
with _sb3:
    _learn_label = "🎓 Hide: Learn About Diabetes" if st.session_state['show_learn'] else "🎓 Learn About Diabetes"
    if st.button(_learn_label, key='btn_learn', use_container_width=True):
        _new = not st.session_state['show_learn']
        st.session_state['show_learn'] = _new
        # Accordion: close the other sections when opening this one
        if _new:
            st.session_state['show_why'] = False
            st.session_state['show_understand'] = False
        st.rerun()

# ── Section 1: Why GlucoFit AI? ──────────────────────────────────────────────
if st.session_state['show_why']:
    st.html("""
    <div class="section features">
      <div class="feat-grid">
        <div class="feat-card">
          <div class="feat-icon">🔍</div>
          <div class="feat-title">Explainable AI (XAI)</div>
          <div class="feat-text">
            SHAP values surface the exact features driving your score —
            no black-box answers, full transparency for patients and clinicians.
          </div>
        </div>
        <div class="feat-card">
          <div class="feat-icon">📊</div>
          <div class="feat-title">Rich Visualisations</div>
          <div class="feat-text">
            Interactive risk gauges, feature-importance charts and trend plots
            make your results instantly understandable at a glance.
          </div>
        </div>
        <div class="feat-card">
          <div class="feat-icon">📥</div>
          <div class="feat-title">Downloadable PDF Report</div>
          <div class="feat-text">
            A branded, doctor-ready PDF summarises your inputs, risk level,
            AI narrative and personalised lifestyle recommendations.
          </div>
        </div>
        <div class="feat-card">
          <div class="feat-icon">🔒</div>
          <div class="feat-title">Privacy First</div>
          <div class="feat-text">
            All processing is local to the session. No data is stored or
            shared — your health information stays entirely yours.
          </div>
        </div>
      </div>
    </div>
    """)

# ── Section 2: Understanding Diabetes ────────────────────────────────────────
if st.session_state['show_understand']:
    st.html("""
    <div class="section visual-section">
      <div class="visual-grid">
        <div class="visual-card">
          <img src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=700&q=80" alt="Blood glucose monitoring" loading="lazy"/>
          <div class="visual-caption">📌 Regular glucose monitoring is the cornerstone of diabetes management.</div>
        </div>
        <div class="visual-card">
          <img src="https://images.unsplash.com/photo-1505576399279-565b52d4ac71?w=700&q=80" alt="Healthy balanced diet" loading="lazy"/>
          <div class="visual-caption">🥗 A balanced, low-glycaemic diet significantly lowers Type-2 diabetes risk.</div>
        </div>
        <div class="visual-card">
          <img src="https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?w=700&q=80" alt="Physical activity and exercise" loading="lazy"/>
          <div class="visual-caption">🏃 30 minutes of daily activity can cut diabetes risk by up to 35%.</div>
        </div>
        <div class="visual-card">
          <img src="https://images.unsplash.com/photo-1532938911079-1b06ac7ceec7?w=700&q=80" alt="Medical data and analytics" loading="lazy"/>
          <div class="visual-caption">📈 AI-powered analytics bring precision to preventive healthcare.</div>
        </div>
        <div class="visual-card">
          <img src="https://images.unsplash.com/photo-1498837167922-ddd27525d352?w=700&q=80" alt="Fresh vegetables and fruits" loading="lazy"/>
          <div class="visual-caption">🍎 Whole foods rich in fibre help regulate blood sugar naturally.</div>
        </div>
      </div>
    </div>
    """)

# ── Section 3: Learn About Diabetes ─────────────────────────────────────────
if st.session_state['show_learn']:
    # ── Video Tabs ──
    _tab_margin_l, _tab_main, _tab_margin_r = st.columns([0.5, 9, 0.5])
    with _tab_main:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌎 Awareness", "🦠 Causes", "🤒 Symptoms", "🛡️ Advice", "🧘 Yoga"])

        with tab1:
            st.html("<br>")
            # Row 1
            _t1_v1, _t1_v2, _t1_v3 = st.columns(3)
            with _t1_v1:
                st.video("https://www.youtube.com/watch?v=t6LaASZHDgc") # 1st Video
            with _t1_v2:
                st.video("https://www.youtube.com/watch?v=wZAjVQWbMlE") # 2nd Video
            with _t1_v3:
                st.video("https://www.youtube.com/watch?v=Hbqe5h72Oxg") # 3rd Video Placeholder

            st.html("<br>")

            # Row 2
            _t1_v4, _t1_v5, _t1_v6 = st.columns(3)
            with _t1_v4:
                st.video("https://youtu.be/4SZGM_E5cLI?si=tpesCLspLdH8aPGc") # 4th Video (Row 2, Left)
            with _t1_v5:
                st.video("https://youtu.be/UDHMN-jQ8kI?si=eDh7ou3tYrgiZHEc") # 5th Video (Row 2, Middle)
            with _t1_v6:
                st.video("https://youtu.be/29fd5I0YMeA?si=0FnGVDYCqOWjqEfy") # 6th Video (Row 2, Right)

        with tab2:
            st.html("<br>")
            _t2_v1, _t2_v2, _t2_v3 = st.columns(3)
            with _t2_v1:
                st.video("https://www.youtube.com/watch?v=XfyGv-xwjlI") # Osmosis HD Explainer
            with _t2_v2:
                st.video("https://youtu.be/wcQCmX8fHW0?si=ip0UlvXgeVa9uWky") # 2nd Video
            with _t2_v3:
                st.video("https://youtu.be/QsSZNetJIuA?si=4lljBVEaIfG2HMw4") # 3rd Video

        with tab3:
            st.html("<br>")
            _t3_v1, _t3_v2, _t3_v3 = st.columns(3)
            with _t3_v1:
                st.video("https://youtu.be/kqaO5JP_roE?si=CD-hGxFF1hMRQ9fF") # Placeholder
            with _t3_v2:
                st.video("https://youtu.be/gGLofmo7q2E?si=FBAFAphiYJ6b2Hrp") # Placeholder
            with _t3_v3:
                pass # Optional 3rd video

        with tab4:
            st.html("<br>")
            _t4_v1, _t4_v2, _t4_v3 = st.columns(3)
            with _t4_v1:
                st.video("https://youtu.be/TLr45n9P4EI?si=EEYVymtj1y7l7Vpw") # Placeholder
            with _t4_v2:
                st.video("https://youtu.be/QyQqr_XtNvg?si=Z1dchetot-gmrmsV") # Placeholder
            with _t4_v3:
                st.video("https://youtu.be/hRwzELaCHDA?si=Ku3vxOKEHFEQbwLc") # Optional 3rd video

        with tab5:

            # ── Yoga category config ──
            _YOGA_CATS = ["🌟 All", "🔥 Pancreatic Health", "⚡ Metabolic Activation", "💨 Pranayama", "🌙 Relaxation"]
            _CAT_HEADERS = {
                "🌟 All":                 ("&#x1F3A5;", "Recommended — Yoga &amp; Diabetes", "background:linear-gradient(135deg,#a78bfa,#60a5fa);"),
                "🔥 Pancreatic Health":   ("&#x1F525;", "Pancreatic &amp; Organ Stimulation", "ycat-accent-fire"),
                "⚡ Metabolic Activation": ("&#x26A1;",  "Circulatory &amp; Metabolic Activation", "ycat-accent-cyan"),
                "💨 Pranayama":           ("&#x1F4A8;", "Pranayama — Breath Control &amp; Oxygenation", "ycat-accent-purple"),
                "🌙 Relaxation":          ("&#x1F319;", "Relaxation &amp; Stress Reduction", "ycat-accent-green"),
            }

            # ── Personalization hint ──
            _suggested = get_personalized_category(dict(st.session_state))
            _default_cat = _suggested if _suggested else "🌟 All"

            st.html("<div style='height:8px'></div>")
            
            yoga_mode = st.radio(
                "Choose Mode:",
                ["🧘 Curated Asanas (Our Library)", "🔍 Search Yoga Videos"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            st.html("<div style='height:16px'></div>")

            yoga_search = ""
            selected_ycat = None

            if yoga_mode == "🔍 Search Yoga Videos":
                # ── Search bar ──
                _ysearch_col1, _ysearch_col2 = st.columns([6, 1])
                with _ysearch_col1:
                    yoga_search = st.text_input(
                        "Search yoga videos",
                        placeholder="🔍  Search e.g. 'pranayama for blood sugar', 'morning yoga routine'...",
                        label_visibility="collapsed",
                        key="yoga_search_input",
                    )
                with _ysearch_col2:
                    if st.button("🔄 Refresh", key="yoga_refresh", use_container_width=True):
                        st.cache_data.clear()

                st.html("<div style='height:16px'></div>")

                # ── Fetch videos ──
                if yoga_search:
                    with st.spinner("Finding yoga videos…"):
                        _num_videos = st.session_state.get("yoga_load_count", 6)
                        videos = fetch_videos(
                            category="🌟 All",
                            search_query=yoga_search,
                            max_results=_num_videos,
                        )
                else:
                    videos = []
                    
            else:
                # ── Category pills for Curated Asanas ──
                selected_ycat = st.pills(
                    "Filter by Asana Category:",
                    _YOGA_CATS,
                    default=_default_cat,
                    label_visibility="collapsed",
                )
                if not selected_ycat:
                    selected_ycat = "🌟 All"

                # ── Personalization banner ──
                if _suggested and selected_ycat == _suggested:
                    st.html(f'''
                    <div style="background:rgba(0,200,170,.08); border:1px solid rgba(0,200,170,.2);
                                border-radius:12px; padding:10px 18px; margin:8px 0 4px;
                                font-size:.82rem; color:#8db8d8; display:flex; align-items:center; gap:8px;">
                        <span style="font-size:1.1rem;">🎯</span>
                        <span>Personalised pick based on your health profile — showing <strong>{_suggested.split(" ", 1)[1]}</strong> videos</span>
                    </div>
                    ''')

                st.html("<div style='height:16px'></div>")

                # ── Fetch videos ──
                with st.spinner("Loading Curated Asanas…"):
                    _num_videos = st.session_state.get("yoga_load_count", 6)
                    videos = fetch_videos(
                        category=selected_ycat,
                        search_query="",
                        max_results=_num_videos,
                        force_fallback=True
                    )

            # ── Search header ──
            if yoga_search:
                st.html(f'''
                <div class="ycat-header">
                  <div class="ycat-accent" style="background:linear-gradient(90deg,#00c8aa,#0091ff);"></div>
                  <span class="ycat-icon">🔍</span>
                  <div><div class="ycat-title">Search results for &ldquo;{html_mod.escape(yoga_search)}&rdquo;</div></div>
                </div>
                ''')

            # ── Render video grid ──
            if not videos:
                st.html('''
                <div class="yv-empty">
                    <span class="yv-empty-icon">🧘</span>
                    No videos found. Try a different search or category.
                </div>
                ''')
            else:
                # Render in rows of 3 — each video embedded inline
                for row_start in range(0, len(videos), 3):
                    row = videos[row_start:row_start + 3]
                    cols = st.columns(3)
                    for i, vid in enumerate(row):
                        with cols[i]:
                            _vid_id = vid["id"]
                            _vid_title = html_mod.escape(vid.get("title", "Yoga Video"))
                            _vid_channel = html_mod.escape(vid.get("channel", ""))
                            _yt_url = f"https://www.youtube.com/watch?v={_vid_id}"

                            st.video(_yt_url)
                            st.html(f'''
                            <div class="vcap">
                              <div class="vcap-name">{_vid_title}</div>
                              <div class="yv-channel" style="padding:0;">{_vid_channel}</div>
                            </div>
                            ''')
                    # Row spacing
                    if row_start + 3 < len(videos):
                        st.html("<div style='height:12px'></div>")

                # ── Load More ──
                st.html("<div style='height:16px'></div>")
                _lm1, _lm2, _lm3 = st.columns([4, 2, 4])
                with _lm2:
                    if st.button("📥 Load More", key="yoga_load_more", use_container_width=True):
                        st.session_state["yoga_load_count"] = _num_videos + 6
                        st.rerun()

st.html("<div style='padding-bottom: 80px;'></div>")


