import streamlit as st

st.set_page_config(
    page_title="How It Works - GlucoFit AI",
    page_icon="📖",
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
    padding: 90px 24px 70px;
    position: relative; z-index: 1;
}

.badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(0,200,170,.12);
    border: 1px solid rgba(0,200,170,.35);
    border-radius: 100px;
    padding: 6px 18px;
    font-family: 'Sora', sans-serif;
    font-size: .78rem; font-weight: 600;
    letter-spacing: .08em; text-transform: uppercase;
    color: #00c8aa;
    margin-bottom: 28px;
    animation: fadeSlideDown .6s ease both;
}

.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: clamp(2.6rem, 6vw, 5rem);
    font-weight: 800;
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
    font-size: clamp(1rem, 2vw, 1.2rem);
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
    display: flex; justify-content: center; gap: 48px; flex-wrap: wrap;
    padding: 36px 24px;
    border-top: 1px solid rgba(255,255,255,.06);
    border-bottom: 1px solid rgba(255,255,255,.06);
    margin: 0 5%;
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
.stat-label { font-size: .82rem; color: #5a7f9a; font-weight: 400; margin-top: 4px; }

/* ══════════════════════════════════════════════════
   PIPELINE / HOW IT WORKS
══════════════════════════════════════════════════ */
.pipeline {
    padding: 45px 5% 60px;
}
.section-heading {
    text-align: center;
    font-family: 'Sora', sans-serif;
    font-size: clamp(1.6rem, 3.5vw, 2.4rem);
    font-weight: 700; color: #ffffff;
    margin-bottom: 10px;
}
.section-sub {
    text-align: center;
    color: #5a7f9a; font-size: .95rem;
    margin-bottom: 52px;
}

.steps-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0;
    position: relative;
}

/* connecting line */
.steps-grid::before {
    content: '';
    position: absolute;
    top: 36px; left: 10%; right: 10%; height: 2px;
    background: linear-gradient(90deg, rgba(0,200,170,.05), rgba(0,200,170,.4), rgba(0,145,255,.4), rgba(0,145,255,.05));
    pointer-events: none;
}

.step-card {
    display: flex; flex-direction: column;
    align-items: center; text-align: center;
    padding: 12px 20px 28px;
    position: relative;
}

.step-icon {
    width: 72px; height: 72px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    margin-bottom: 20px;
    position: relative; z-index: 1;
}

.step-icon.c1 { background: rgba(0,200,170,.15); border: 2px solid rgba(0,200,170,.4); }
.step-icon.c2 { background: rgba(0,160,200,.15); border: 2px solid rgba(0,160,200,.4); }
.step-icon.c3 { background: rgba(0,120,230,.15); border: 2px solid rgba(0,120,230,.4); }
.step-icon.c4 { background: rgba(90,80,230,.15);  border: 2px solid rgba(90,80,230,.4); }
.step-icon.c5 { background: rgba(160,60,200,.15); border: 2px solid rgba(160,60,200,.4); }

.step-num {
    position: absolute; top: -6px; right: -6px;
    width: 22px; height: 22px; border-radius: 50%;
    background: linear-gradient(135deg, #00c8aa, #0091ff);
    color: #fff; font-family: 'Sora', sans-serif;
    font-size: .65rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}

.step-title {
    font-family: 'Sora', sans-serif;
    font-size: .9rem; font-weight: 600; color: #e0f0fc;
    margin-bottom: 8px;
}
.step-desc { font-size: .8rem; color: #5a7f9a; line-height: 1.5; }

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
</style>
""")

# ── Navigation Home ──
st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
_back_l, _back_r = st.columns([1.5, 9.5])
with _back_l:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")

# ══════════════════════════════════════════════════════════════════════════════
# HOW IT WORKS – 5-step pipeline
# ══════════════════════════════════════════════════════════════════════════════
st.html("""
<div class="section pipeline" id="how-it-works">
  <h2 class="section-heading">How GlucoFit AI Works</h2>
  <p class="section-sub">A guided 5-step journey from your data to actionable insights</p>

  <div class="steps-grid">

    <div class="step-card">
      <div class="step-icon c1">
        👤
        <div class="step-num">1</div>
      </div>
      <div class="step-title">Personal Information</div>
      <div class="step-desc">Age, gender, ethnicity & family history — the demographic foundation.</div>
    </div>

    <div class="step-card">
      <div class="step-icon c2">
        🧪
        <div class="step-num">2</div>
      </div>
      <div class="step-title">Clinical Inputs</div>
      <div class="step-desc">Glucose, HbA1c, BMI, blood pressure & cholesterol readings.</div>
    </div>

    <div class="step-card">
      <div class="step-icon c3">
        🥗
        <div class="step-num">3</div>
      </div>
      <div class="step-title">Lifestyle & Diet</div>
      <div class="step-desc">Activity levels, dietary patterns, sleep quality & smoking status.</div>
    </div>

    <div class="step-card">
      <div class="step-icon c4">
        🔮
        <div class="step-num">4</div>
      </div>
      <div class="step-title">Prediction</div>
      <div class="step-desc">Real-time ML inference with risk score, probability & key drivers.</div>
    </div>

    <div class="step-card">
      <div class="step-icon c5">
        📄
        <div class="step-num">5</div>
      </div>
      <div class="step-title">AI Analysis & Report</div>
      <div class="step-desc">Personalised insights, lifestyle recommendations & downloadable PDF.</div>
    </div>

  </div>
</div>
""")

# ── CTA button after pipeline ────────────────────────────────────────────────
_pl, _pc, _pr = st.columns([2, 2, 2])
with _pc:
    if st.button("🚀  Begin Your Assessment", use_container_width=True, key='pipeline_cta'):
        st.switch_page("pages/2_Assessment.py")

# Add some spacing at the bottom
st.html("<br><br><br>")