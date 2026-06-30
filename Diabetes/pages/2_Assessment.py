import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from google import genai
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
from typing import List, Dict, Any
import itertools
from nutrition_analyzer import (
    fetch_nutrition, classify_diet,
    diet_type_to_model_value, diet_type_to_lifestyle_label,
    manual_label_to_classifier_label,
)

def get_gemini_api_keys() -> List[str]:
    keys = []
    try:
        if "GEMINI_API_KEYS" in st.secrets:
            val = st.secrets["GEMINI_API_KEYS"]
            if isinstance(val, list): keys.extend(val)
            elif isinstance(val, str): keys.extend([k.strip() for k in val.split(",") if k.strip()])
        elif "GEMINI_API_KEY" in st.secrets:
            keys.append(st.secrets["GEMINI_API_KEY"])
    except Exception: pass
    return keys

@st.cache_resource
def load_model(): return joblib.load('diabetes_model.pkl')
@st.cache_resource
def load_scaler(): return joblib.load('scaler.pkl')
@st.cache_data
def load_diabetes_data():
    df = pd.read_csv('diabetes.csv')
    def _assign_diet(row):
        g, b, i = row['Glucose'], row['BMI'], row['Insulin']
        if g > 140: return 0
        if b < 25 and g <= 110: return 1
        if b >= 33 and g <= 140: return 2
        return 3
    df['DietType'] = df.apply(_assign_diet, axis=1)
    return df

def lifestyle_risk_delta(lifestyle: dict) -> float:
    delta = 0.0
    delta += {'Never':+0.10,'1-2 days':+0.05,'3-4 days':0.00,'5-6 days':-0.04,'Daily':-0.06}.get(lifestyle.get('exercise_frequency','3-4 days'),0.0)
    delta += {'High Carbs':+0.08,'High Fat':+0.06,'Low Protein':+0.03,'Balanced':0.00}.get(lifestyle.get('diet_type','Balanced'),0.0)
    delta += {'Very High':+0.08,'High':+0.05,'Moderate':0.00,'Low':-0.02,'Very Low':-0.03}.get(lifestyle.get('stress_level','Moderate'),0.0)
    sleep = lifestyle.get('sleep_hours', 7)
    if sleep <= 5: delta += 0.07
    elif sleep == 6: delta += 0.03
    elif sleep >= 9: delta += 0.02
    delta += {'Never':0.00,'Occasionally':+0.02,'Regularly':+0.07}.get(lifestyle.get('alcohol_consumption','Never'),0.0)
    delta += {'Never':0.00,'Former':+0.02,'Current':+0.08}.get(lifestyle.get('smoking','Never'),0.0)
    return delta

model = load_model()
scaler = load_scaler()

st.set_page_config(
    page_title="Assessment – GlucoFit AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    /* Hide Streamlit chrome */
    #MainMenu, footer, header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"]         { display: none !important; }

    .main { background: linear-gradient(135deg, #0a1628 0%, #1a3a52 50%, #0d2340 100%); background-attachment: fixed; }
    .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; padding-left: 2rem !important; padding-right: 2rem !important; max-width: 960px !important; margin-left: auto !important; margin-right: auto !important; }
    .header-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 14px 20px; border-radius: 12px; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6); margin-bottom: 12px; margin-top: 0; width: 100%; box-sizing: border-box; animation: fadeInDown 0.8s ease-out; border-left: 5px solid #ff006e; }
    .header { text-align: center; color: #ffffff; font-size: 1.8em; margin: 0; font-weight: bold; text-shadow: 0 4px 8px rgba(0,0,0,0.4); letter-spacing: 1px; }
    .subheader { text-align: center; color: #a8b5ff; font-size: 0.85em; margin: 4px 0 0 0; font-weight: 500; letter-spacing: 0.5px; }
    .stButton > button { font-size: 0.95em; min-height: 36px; border-radius: 8px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: 2px solid rgba(255, 0, 110, 0.5); }
    .stButton > button:hover { background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); border: 2px solid #ff006e; }
    .input-section { background: linear-gradient(135deg, rgba(26, 58, 82, 0.7) 0%, rgba(13, 35, 64, 0.7) 100%); color: #ffffff; padding: 12px; border-radius: 10px; margin: 8px 0; border: 2px solid rgba(102, 126, 234, 0.3); backdrop-filter: blur(10px); }
    @keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
    [data-testid="stVerticalBlock"] > [style*="flex-direction"] > [data-testid="stVerticalBlock"] { margin-bottom: 0.3rem !important; }
    @media (max-width: 768px) { .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; padding-top: 0.5rem !important; } .header { font-size: 1.2em !important; } .subheader { font-size: 0.75em !important; } .input-section { padding: 10px 10px !important; } .stButton > button { font-size: 0.82em !important; min-height: 34px; } }
    @media (min-width: 769px) and (max-width: 1024px) { .block-container { padding-left: 1.25rem !important; padding-right: 1.25rem !important; } }
    </style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state: st.session_state.page = 1

# Scroll to top on every page change
import streamlit.components.v1 as components
components.html(f"<script>window.parent.document.querySelector('section.main').scrollTo(0,0);</script><span id='scroll_page_{st.session_state.page}'></span>", height=0)

st.markdown('<div class="header-container"><div class="header">🏥 GlucoFit AI</div><div class="subheader">💙 Your Health Matters - Get Instant Predictions</div></div>', unsafe_allow_html=True)
if 'personal_data' not in st.session_state: st.session_state.personal_data = {}
if 'clinical_data' not in st.session_state: st.session_state.clinical_data = {}
if 'lifestyle_data' not in st.session_state: st.session_state.lifestyle_data = {}
if 'nutrition_analysis' not in st.session_state: st.session_state.nutrition_analysis = None
if 'ai_advice' not in st.session_state: st.session_state.ai_advice = None
if 'insulin_estimated' not in st.session_state: st.session_state.insulin_estimated = False

# ── Insulin Resistance Estimation Questionnaire (Modal) ──────────────────
@st.dialog("💉 Estimate Your Insulin Level")
def insulin_questionnaire():
    st.caption("Answer 4 simple questions. Your insulin index will be calculated automatically.")
    st.markdown("**1. Where does your body store most of its fat?**")
    q1 = st.radio("q1", ["Hips or Thighs", "Evenly all over", "Mostly in the belly"], key="iq_fat", horizontal=True, label_visibility="collapsed")
    st.markdown("**2. Do you have dark, velvety patches on your skin** *(neck, underarms, knees)*?")
    q2 = st.radio("q2", ["No", "A few, faint ones", "Yes, clearly visible"], key="iq_skin", horizontal=True, label_visibility="collapsed")
    st.markdown("**3. How do you feel 30–60 minutes after eating a big meal?**")
    q3 = st.radio("q3", ["Fine / Normal", "Slightly tired", "Very sleepy or sluggish"], key="iq_energy", horizontal=True, label_visibility="collapsed")
    st.markdown("**4. How often do you feel a strong urge to eat sweets or sugary snacks?**")
    q4 = st.radio("q4", ["Rarely or never", "Sometimes", "Very often / Always"], key="iq_cravings", horizontal=True, label_visibility="collapsed")
    st.divider()
    w1 = {"Hips or Thighs": 0, "Evenly all over": 2, "Mostly in the belly": 3}[q1]
    w2 = {"No": 0, "A few, faint ones": 3, "Yes, clearly visible": 5}[q2]
    w3 = {"Fine / Normal": 0, "Slightly tired": 2, "Very sleepy or sluggish": 4}[q3]
    w4 = {"Rarely or never": 0, "Sometimes": 1, "Very often / Always": 2}[q4]
    total_score = w1 + w2 + w3 + w4
    insulin_index = round((total_score / 14) * 100, 1)
    if insulin_index <= 30: risk_label, risk_color = "🟢 Low — Insulin Sensitive", "#55efc4"
    elif insulin_index <= 65: risk_label, risk_color = "🟡 Moderate — Early Resistance", "#fdcb6e"
    else: risk_label, risk_color = "🔴 High — Likely Insulin Resistant", "#ff7675"
    st.markdown(f"""<div style="background:rgba(26,58,82,0.6);border-radius:10px;padding:14px 16px;text-align:center;border:1.5px solid rgba(102,126,234,0.45);margin-bottom:10px;">
    <div style="font-size:0.75em;color:#a8b5ff;margin-bottom:6px;">Estimated Insulin Index (0–100)</div>
    <div style="font-size:2.4em;font-weight:700;color:#fff;line-height:1.1;">{insulin_index}</div>
    <div style="font-size:0.88em;font-weight:600;color:{risk_color};margin-top:6px;">{risk_label}</div></div>""", unsafe_allow_html=True)
    if st.button("✅ Apply This Value", use_container_width=True, type="primary"):
        _rounded = int(round(insulin_index))
        st.session_state.clinical_data['insulin'] = _rounded
        st.session_state.insulin_estimated = True
        st.session_state._insulin_risk_label = risk_label
        st.rerun()

# ── Local advice fallback (used when Gemini API is unavailable) ──────────
def _generate_local_advice(session, result_str, confidence, top_risks):
    personal = session.personal_data
    clinical = session.clinical_data
    lifestyle = session.lifestyle_data
    name = personal.get('name', 'User')
    glucose = clinical.get('glucose', 100)
    bmi = clinical.get('bmi', 25.0)
    bp = clinical.get('blood_pressure', 70)
    dpf = clinical.get('diabetes_pedigree', 0.078)
    exercise = lifestyle.get('exercise_frequency', '3-4 days')
    sleep = lifestyle.get('sleep_hours', 7)
    stress = lifestyle.get('stress_level', 'Moderate')
    diet_type = lifestyle.get('diet_type', 'Balanced')

    risk_lines = []
    if glucose > 140:
        risk_lines.append(f"- ⚠️ Your **glucose ({glucose} mg/dL)** is above normal range (70–100 mg/dL fasting). This is a significant risk factor.")
    elif glucose > 100:
        risk_lines.append(f"- 🟡 Your **glucose ({glucose} mg/dL)** is slightly elevated. Normal fasting is 70–100 mg/dL.")
    else:
        risk_lines.append(f"- ✅ Your **glucose ({glucose} mg/dL)** is within normal range.")
    if bmi >= 30:
        risk_lines.append(f"- ⚠️ Your **BMI ({bmi:.1f})** falls in the obese category (≥30). Weight management is strongly recommended.")
    elif bmi >= 25:
        risk_lines.append(f"- 🟡 Your **BMI ({bmi:.1f})** is in the overweight range (25–29.9).")
    else:
        risk_lines.append(f"- ✅ Your **BMI ({bmi:.1f})** is healthy.")
    if dpf > 0.5:
        risk_lines.append(f"- ⚠️ Your **Diabetes Pedigree Function ({dpf:.3f})** indicates significant family history.")
    if bp > 80:
        risk_lines.append(f"- 🟡 Your **blood pressure ({bp} mm Hg diastolic)** is above optimal (<80).")
    risk_text = '\n'.join(risk_lines)

    risk_factor_text = ""
    if top_risks:
        risk_factor_text = "\n**Top contributing risk factors:** " + ", ".join([f"{r['name']} ({r['value']:.1f})" for r in top_risks])

    diet_advice = {
        'High Carbs': "Your diet is **high in carbohydrates**. Consider reducing refined carbs (white rice, sugar, maida) and increasing fiber-rich whole grains, vegetables, and protein.",
        'High Fat': "Your diet is **high in fat**. Reduce fried foods, ghee, and full-fat dairy. Opt for lean proteins, steamed/grilled preparations, and healthy fats like nuts.",
        'Balanced': "Your diet appears **balanced** — keep it up! Focus on portion control and include plenty of vegetables, lean protein, and whole grains.",
        'Low Protein': "Your diet is **low in protein**. Include more dal, paneer, eggs, chicken, or legumes to support muscle health and blood sugar stability.",
    }.get(diet_type, "Maintain a balanced diet with vegetables, lean protein, and whole grains.")

    exercise_advice = {
        'Never': "You reported **no exercise**. Start with 15–20 min brisk walking daily and gradually increase to 150 min/week.",
        '1-2 days': "You exercise **1–2 days/week**. Try to increase to at least 3–4 days with 30 min of moderate activity.",
        '3-4 days': "Good — you exercise **3–4 days/week**. Maintain this and consider adding strength training.",
        '5-6 days': "Excellent — **5–6 days/week** of exercise is great for glucose control. Keep it up!",
        'Daily': "Outstanding — **daily exercise** significantly lowers diabetes risk. Maintain your routine!",
    }.get(exercise, "Aim for at least 150 minutes of moderate exercise per week.")

    sleep_note = f"You sleep **{sleep} hours/night**. "
    if sleep < 6:
        sleep_note += "This is below recommended (7–9 hrs). Poor sleep raises cortisol and blood sugar."
    elif sleep > 9:
        sleep_note += "Sleeping over 9 hours may also impact metabolic health. Aim for 7–8 hours."
    else:
        sleep_note += "This is within the healthy range (7–9 hrs). Good job!"

    stress_note = f"Your stress level is **{stress}**. "
    if stress in ('High', 'Very High'):
        stress_note += "High stress increases cortisol, which raises blood sugar. Try deep breathing, meditation, or yoga."
    else:
        stress_note += "Keep managing stress with regular breaks, hobbies, and physical activity."

    advice = f"""### ⚠️ Clinical Risk Assessment

**{name}**, your prediction is **{result_str}** with **{confidence:.1f}%** confidence.

{risk_text}
{risk_factor_text}

### 🥑 Personalized Nutrition Plan

{diet_advice}

**Suggested daily targets:**
- 🔥 Calories: {1800 if bmi < 25 else 1500}–{2200 if bmi < 25 else 1800} kcal
- 🍞 Carbs: 45–55% (prefer complex carbs)
- 🥩 Protein: 20–30%
- 🧈 Fat: 20–25% (healthy fats)

**Meal ideas:**
- 🌅 Breakfast: Oats with nuts, or 2 moong dal chilla with mint chutney
- ☀️ Lunch: 1 cup brown rice, dal, sabzi, and salad
- 🌙 Dinner: 2 multigrain roti with paneer/chicken and vegetables

### 🏃 Exercise & Activity Protocol

{exercise_advice}

**Recommended activities:**
- 🚶 30 min brisk walking (5 days/week)
- 🧘 Yoga or stretching (2–3 days/week)
- 💪 Light resistance/strength training (2 days/week)

### 🧘 Stress & Sleep Optimization

{sleep_note}

{stress_note}

**Tips:**
- Practice 10 min meditation or deep breathing daily
- Maintain a consistent sleep schedule
- Limit screen time 1 hour before bed

### 📊 Monitoring Targets

- 🩸 **Fasting glucose target:** 70–100 mg/dL
- 🍽️ **Post-meal glucose (2 hrs):** <140 mg/dL
- ⚖️ **BMI target:** 18.5–24.9
- 🏥 **Get HbA1c tested** every 3–6 months
- 📅 Schedule a doctor visit if glucose stays above 126 mg/dL fasting

### 🖼️ Recommended Visuals
Healthy Eating Plate
Walking Exercise
Blood Glucose Monitor
Yoga and Stretching
Sleep Quality"""
    return advice

# Page 1: Personal Information
if st.session_state.page == 1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("📋  Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input('👤 Full Name', value=st.session_state.personal_data.get('name', ''), key='p1_name')
        age = st.number_input('🎂 Age (years)', min_value=1, max_value=120,
                              value=int(st.session_state.personal_data.get('age', 30)),
                              step=1, key='p1_age')
        gender = st.radio('👫 Gender', ['Male', 'Female'],
                          index=0 if st.session_state.personal_data.get('gender', 'Male') == 'Male' else 1,
                          key='p1_gender')
    with col2:
        email = st.text_input('📧 Email (Optional)', value=st.session_state.personal_data.get('email', ''), key='p1_email')
        pregnancies = st.number_input('🤰 Number of Pregnancies', min_value=0, max_value=20,
                                      value=int(st.session_state.personal_data.get('pregnancies', 0)),
                                      step=1, disabled=(gender != 'Female'), key='pregnancies_input')
        if gender == 'Male': pregnancies = 0
    st.session_state.personal_data = {'name': name, 'age': int(age), 'gender': gender, 'email': email, 'pregnancies': int(pregnancies)}
    st.markdown('</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button('🏠 Home', use_container_width=True, key='page1_home'):
            st.switch_page('app.py')
    with col3:
        if st.button('Next →', use_container_width=True, key='page1_next'):
            st.session_state.page = 2; st.rerun()

# Page 2: Clinical Data
elif st.session_state.page == 2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("⚕️ Clinical Data")
    col1, col2 = st.columns(2)
    with col1:
        glucose = st.number_input('🍬 Glucose Level (mg/dL)', min_value=0, max_value=200, value=st.session_state.clinical_data.get('glucose', 100), step=1, key='glucose', help='Normal range: 70–100 mg/dL (fasting)')
        blood_pressure = st.number_input('❤️ Diastolic Blood Pressure (mm Hg)', min_value=0, max_value=150, value=st.session_state.clinical_data.get('blood_pressure', 70), step=1, key='bp', help='Normal diastolic: 60–80 mm Hg')
        _skin_options = ['Slim (7–22 mm)', 'Average (23–36 mm)', 'Overweight (37–99 mm)']
        _skin_map = {'Slim (7–22 mm)': 17, 'Average (23–36 mm)': 29, 'Overweight (37–99 mm)': 43}
        _skin_reverse = {v: k for k, v in _skin_map.items()}
        _skin_default_label = _skin_reverse.get(st.session_state.clinical_data.get('skin_thickness', 29), 'Average (23–36 mm)')
        _skin_idx = _skin_options.index(_skin_default_label) if _skin_default_label in _skin_options else 1
        skin_body_type = st.selectbox('👤 Body Type (Skin Thickness)', options=_skin_options, index=_skin_idx, key='skin', help='Select your body type.')
        skin_thickness = _skin_map[skin_body_type]
        # Insulin — set via questionnaire only
        insulin = st.session_state.clinical_data.get('insulin', 0)
        _ir_estimated = st.session_state.get('insulin_estimated', False)
        _ir_risk = st.session_state.get('_insulin_risk_label', '')
        if _ir_estimated:
            st.markdown(f'<div style="background:rgba(26,58,82,0.55);border:1.5px solid rgba(102,126,234,0.4);border-radius:10px;padding:10px 14px;margin-bottom:6px;"><div style="font-size:0.75em;color:#a8b5ff;margin-bottom:2px;">💉 Insulin Index (Estimated)</div><div style="font-size:1.8em;font-weight:700;color:#fff;">{insulin}</div><div style="font-size:0.78em;color:#a8b5ff;margin-top:2px;">{_ir_risk}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:rgba(26,58,82,0.35);border:1.5px dashed rgba(102,126,234,0.3);border-radius:10px;padding:10px 14px;margin-bottom:6px;text-align:center;"><div style="font-size:0.82em;color:#6c757d;">💉 Insulin value not set yet</div></div>', unsafe_allow_html=True)
        if st.button('📝 Add Insulin (Estimate)', use_container_width=True, key='open_insulin_q'):
            insulin_questionnaire()
    with col2:
        st.markdown("**⚖️ BMI Calculator**", help='Normal BMI: 18.5–24.9. Overweight: 25–29.9. Obese: 30+.')
        height_unit = st.radio("Height unit", ["cm", "ft / in"], index=0 if st.session_state.clinical_data.get('height_unit', 'cm') == 'cm' else 1, horizontal=True, key='height_unit')
        if height_unit == "cm":
            height_cm = st.number_input('📏 Height (cm)', min_value=50.0, max_value=250.0, value=float(st.session_state.clinical_data.get('height_cm', 170.0)), step=0.5, key='height_cm')
            height_m = height_cm / 100.0
        else:
            _ft_col, _in_col = st.columns(2)
            with _ft_col: height_ft = st.number_input('Feet', min_value=1, max_value=8, value=int(st.session_state.clinical_data.get('height_ft', 5)), step=1, key='height_ft')
            with _in_col: height_in = st.number_input('Inches', min_value=0, max_value=11, value=int(st.session_state.clinical_data.get('height_in', 7)), step=1, key='height_in')
            height_m = (height_ft * 12 + height_in) * 0.0254
            height_cm = round(height_m * 100, 1)
        weight_kg = st.number_input('⚖️ Weight (kg)', min_value=10.0, max_value=300.0, value=float(st.session_state.clinical_data.get('weight_kg', 70.0)), step=0.5, key='weight_kg')
        bmi = round(weight_kg / (height_m ** 2), 1) if height_m > 0 else 0.0
        if bmi < 18.5: _bmi_cat, _bmi_color = "Underweight", "#74b9ff"
        elif bmi < 25.0: _bmi_cat, _bmi_color = "Normal ✅", "#55efc4"
        elif bmi < 30.0: _bmi_cat, _bmi_color = "Overweight ⚠️", "#fdcb6e"
        else: _bmi_cat, _bmi_color = "Obese 🔴", "#ff7675"
        st.markdown(f"""<div style="background:rgba(102,126,234,0.15);border:1.5px solid rgba(102,126,234,0.5);border-radius:10px;padding:10px 16px;margin-top:6px;text-align:center;">
        <div style="font-size:0.78em;color:#a8b5ff;margin-bottom:2px;">Calculated BMI</div>
        <div style="font-size:2.0em;font-weight:700;color:#ffffff;line-height:1.1;">{bmi:.1f}</div>
        <div style="font-size:0.82em;font-weight:600;color:{_bmi_color};margin-top:2px;">{_bmi_cat}</div></div>""", unsafe_allow_html=True)

    # Family History → DPF
    st.markdown("---")
    st.subheader("👨‍👩‍👧 Family History (Diabetes Pedigree Function)")
    st.caption("Select which family members have been diagnosed with diabetes.")
    fh_col1, fh_col2 = st.columns(2)
    with fh_col1:
        fh_mother = st.checkbox("👩 Mother has diabetes", value=st.session_state.clinical_data.get('fh_mother', False), key='fh_mother')
        fh_father = st.checkbox("👨 Father has diabetes", value=st.session_state.clinical_data.get('fh_father', False), key='fh_father')
    with fh_col2:
        fh_sibling = st.checkbox("👫 Sibling(s) have diabetes", value=st.session_state.clinical_data.get('fh_sibling', False), key='fh_sibling')
        fh_grandparent = st.checkbox("👴 Grandparent(s) have diabetes", value=st.session_state.clinical_data.get('fh_grandparent', False), key='fh_grandparent')
    any_family = fh_mother or fh_father or fh_sibling or fh_grandparent
    _DPF_MIN, _DPF_MAX = 0.078, 2.42
    if any_family:
        fh_col3, fh_col4 = st.columns(2)
        with fh_col3:
            onset_options = ['Unknown', 'Before age 30', 'Age 30–50', 'After age 50']
            onset_age = st.selectbox("📅 Age of onset in family", onset_options, index=onset_options.index(st.session_state.clinical_data.get('fh_onset', 'Unknown')), key='fh_onset')
        with fh_col4:
            member_options = ['1', '2', '3', '4+']
            num_affected = st.selectbox("🔢 Total affected members", member_options, index=member_options.index(st.session_state.clinical_data.get('fh_count', '1')), key='fh_count')
        base_score = 0.35*int(fh_mother) + 0.35*int(fh_father) + 0.25*int(fh_sibling) + 0.15*int(fh_grandparent)
        onset_multiplier = {'Before age 30': 1.5, 'Age 30–50': 1.0, 'After age 50': 0.7, 'Unknown': 1.0}[onset_age]
        member_scale = {'1': 1.0, '2': 1.3, '3': 1.6, '4+': 2.0}[num_affected]
        diabetes_pedigree = round(min(max(base_score * onset_multiplier * member_scale, _DPF_MIN), _DPF_MAX), 3)
        st.metric("Calculated DPF", f"{diabetes_pedigree:.3f}")
        st.progress(min(diabetes_pedigree / _DPF_MAX, 1.0))
        if diabetes_pedigree < 0.268: st.success("🟢 **Low genetic risk**")
        elif diabetes_pedigree < 0.526: st.warning("🟡 **Moderate genetic risk**")
        else: st.error("🔴 **High genetic risk**")
    else:
        onset_age = 'Unknown'; num_affected = '1'
        diabetes_pedigree = _DPF_MIN
        st.info("ℹ️ **No family history selected** — DPF set to minimum (0.078).")

    st.session_state.clinical_data = {
        'glucose': glucose, 'blood_pressure': blood_pressure, 'skin_thickness': skin_thickness,
        'skin_body_type': skin_body_type, 'insulin': insulin, 'bmi': bmi,
        'diabetes_pedigree': diabetes_pedigree, 'fh_mother': fh_mother, 'fh_father': fh_father,
        'fh_sibling': fh_sibling, 'fh_grandparent': fh_grandparent,
        'fh_onset': onset_age if any_family else 'Unknown',
        'fh_count': num_affected if any_family else '1',
        'height_unit': height_unit, 'height_cm': height_cm, 'weight_kg': weight_kg,
        'height_ft': int(st.session_state.get('height_ft', 5)) if height_unit != 'cm' else 5,
        'height_in': int(st.session_state.get('height_in', 7)) if height_unit != 'cm' else 7,
    }
    st.markdown('</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button('← Back', use_container_width=True, key='page2_back'): st.session_state.page = 1; st.rerun()
    with col3:
        if st.button('Next →', use_container_width=True, key='page2_next'): st.session_state.page = 3; st.rerun()

# Page 3: Lifestyle Factors
elif st.session_state.page == 3:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("🏃 Lifestyle Factors")
    col1, col2 = st.columns(2)
    with col1:
        exercise_frequency = st.select_slider('🏋️ Exercise Frequency per week', options=['Never', '1-2 days', '3-4 days', '5-6 days', 'Daily'], value=st.session_state.lifestyle_data.get('exercise_frequency', '3-4 days'), key='exercise')
        sleep_hours = st.slider('😴 Average Sleep Hours per Night', min_value=4, max_value=12, value=st.session_state.lifestyle_data.get('sleep_hours', 7), step=1, key='sleep')
    with col2:
        stress_level = st.select_slider('😰 Stress Level', options=['Very Low', 'Low', 'Moderate', 'High', 'Very High'], value=st.session_state.lifestyle_data.get('stress_level', 'Moderate'), key='stress')
    col3, col4 = st.columns(2)
    with col3:
        alcohol_consumption = st.radio('🍷 Alcohol Consumption', ['Never', 'Occasionally', 'Regularly'], index=['Never', 'Occasionally', 'Regularly'].index(st.session_state.lifestyle_data.get('alcohol_consumption', 'Never')), key='alcohol')
    with col4:
        smoking = st.radio('🚭 Smoking Status', ['Never', 'Former', 'Current'], index=['Never', 'Former', 'Current'].index(st.session_state.lifestyle_data.get('smoking', 'Never')), key='smoking')
    st.markdown('</div>', unsafe_allow_html=True)

    # Diet Analysis
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("🥗 Daily Diet Analysis")
    st.caption("Enter what you ate. Separate items with **'and'**.")
    _mc1, _mc2, _mc3 = st.columns(3)
    with _mc1: breakfast_input = st.text_area('🌅 Breakfast', value=st.session_state.lifestyle_data.get('breakfast_input', ''), placeholder='e.g. 2 chapati and chai', height=110, key='breakfast_input_area')
    with _mc2: lunch_input = st.text_area('☀️ Lunch', value=st.session_state.lifestyle_data.get('lunch_input', ''), placeholder='e.g. 1 cup rice and dal', height=110, key='lunch_input_area')
    with _mc3: dinner_input = st.text_area('🌙 Dinner', value=st.session_state.lifestyle_data.get('dinner_input', ''), placeholder='e.g. paneer and 2 roti', height=110, key='dinner_input_area')
    _meal_parts = [m.strip() for m in [breakfast_input, lunch_input, dinner_input] if m.strip()]
    food_input = ' and '.join(_meal_parts)
    if food_input != st.session_state.lifestyle_data.get('food_input', ''): st.session_state.nutrition_analysis = None
    _edamam_app_id = _edamam_app_key = None
    try: _edamam_app_id = st.secrets.get('EDAMAM_APP_ID', None); _edamam_app_key = st.secrets.get('EDAMAM_APP_KEY', None)
    except Exception: pass
    _ac1, _ac2 = st.columns([1, 3])
    with _ac1: analyse_clicked = st.button('🔍 Analyse Diet', use_container_width=True, key='analyse_diet')
    if analyse_clicked and _meal_parts:
        with st.spinner('Analysing nutrition…'):
            raw = fetch_nutrition(food_input, _edamam_app_id, _edamam_app_key)
        if raw and raw.get('calories', 0) > 0:
            tc, tp, tf = raw['total_carbs'], raw['total_protein'], raw['total_fat']
            result = classify_diet(tc, tp, tf)
            result['calories'] = raw.get('calories', 0)
            result['total_carbs'] = tc; result['total_protein'] = tp; result['total_fat'] = tf
            result['items'] = raw.get('items', [])
            st.session_state.nutrition_analysis = result
        else:
            unrecognized = raw.get('errors', []) if raw else []
            if unrecognized:
                st.warning('⚠️ Could not estimate: ' + ', '.join(unrecognized))
            else:
                st.warning('⚠️ No food items recognized. Try: "2 chapati and dal and paneer"')
    _nutrition = st.session_state.nutrition_analysis
    if _nutrition and _nutrition.get('calories', 0) > 0:
        st.markdown('---')
        st.markdown('#### 📊 Nutrition Analysis Results')
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric('🔥 Calories', f"{_nutrition['calories']:.0f} kcal")
        _m2.metric('🍞 Carbs', f"{_nutrition['total_carbs']:.1f} g", delta=f"{_nutrition.get('carb_ratio',0)*100:.0f}%", delta_color='off')
        _m3.metric('🥩 Protein', f"{_nutrition['total_protein']:.1f} g", delta=f"{_nutrition.get('protein_ratio',0)*100:.0f}%", delta_color='off')
        _m4.metric('🧈 Fat', f"{_nutrition['total_fat']:.1f} g", delta=f"{_nutrition.get('fat_ratio',0)*100:.0f}%", delta_color='off')
        _dtype = _nutrition['diet_type']
        _badge_colors = {'Carbohydrate Rich Diet': ('#fdcb6e', '#2d3436'), 'Protein Rich Diet': ('#55efc4', '#2d3436'), 'Fat Rich Diet': ('#ff7675', '#ffffff'), 'Balanced Diet': ('#74b9ff', '#2d3436')}
        _bg, _fg = _badge_colors.get(_dtype, ('#74b9ff', '#2d3436'))
        _src = '🌐 via Edamam API' if _edamam_app_id and any('(est.)' not in str(r) for r in _nutrition.get('items', [])) else '📖 via Built-in DB'
        st.markdown(f'<div style="text-align:center;margin:12px 0;"><span style="background:{_bg};color:{_fg};padding:8px 24px;border-radius:20px;font-weight:700;font-size:1.1em;">{_dtype}</span>&nbsp;<span style="font-size:0.75em;color:#a8b5ff;">{_src}</span></div>', unsafe_allow_html=True)
        if _nutrition.get('items'):
            with st.expander('🔎 Per-item calorie breakdown', expanded=True):
                _items_df = pd.DataFrame(_nutrition['items'])
                _items_df.columns = ['Food Item', 'Weight (g)', 'Calories', 'Carbs (g)', 'Protein (g)', 'Fat (g)']
                st.dataframe(_items_df, use_container_width=True, hide_index=True)
        diet_type = diet_type_to_lifestyle_label(_dtype)
    else:
        diet_type = st.session_state.lifestyle_data.get('diet_type', 'Balanced')
    st.markdown('</div>', unsafe_allow_html=True)
    st.session_state.lifestyle_data = {'exercise_frequency': exercise_frequency, 'sleep_hours': sleep_hours, 'diet_type': diet_type, 'stress_level': stress_level, 'alcohol_consumption': alcohol_consumption, 'smoking': smoking, 'breakfast_input': breakfast_input, 'lunch_input': lunch_input, 'dinner_input': dinner_input, 'food_input': food_input}
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button('← Back', use_container_width=True, key='page3_back'): st.session_state.page = 2; st.rerun()
    with col3:
        if st.button('Get Prediction →', use_container_width=True, key='page3_next'): st.session_state.page = 4; st.rerun()

# Page 4: Prediction Result
elif st.session_state.page == 4:
    _p4_diet_label = st.session_state.lifestyle_data.get('diet_type', 'Balanced')
    _p4_classifier_label = manual_label_to_classifier_label(_p4_diet_label)
    _p4_diet_encoded = diet_type_to_model_value(_p4_classifier_label)
    input_data = np.array([[st.session_state.personal_data.get('pregnancies', 0), st.session_state.clinical_data.get('glucose', 100), st.session_state.clinical_data.get('blood_pressure', 70), st.session_state.clinical_data.get('skin_thickness', 20), st.session_state.clinical_data.get('insulin', 80), st.session_state.clinical_data.get('bmi', 25.0), st.session_state.clinical_data.get('diabetes_pedigree', 0.5), st.session_state.personal_data.get('age', 30), _p4_diet_encoded]])
    std_data = scaler.transform(input_data)
    prediction = model.predict(std_data)
    prediction_proba = model.predict_proba(std_data)
    _lifestyle = st.session_state.lifestyle_data
    _delta = lifestyle_risk_delta(_lifestyle)
    _raw_diabetic_prob = float(prediction_proba[0][1])
    _adj_diabetic_prob = float(np.clip(_raw_diabetic_prob + _delta, 0.01, 0.99))
    _adj_non_diabetic_prob = 1.0 - _adj_diabetic_prob
    _adj_prediction = 1 if _adj_diabetic_prob >= 0.5 else 0
    prediction = np.array([_adj_prediction])
    prediction_proba = np.array([[_adj_non_diabetic_prob, _adj_diabetic_prob]])

    st.divider()
    st.subheader("📊 Prediction Results")
    if prediction[0] == 0:
        st.success('✅ Great News!', icon="✨")
        st.markdown(f"### 🎉 Hello {st.session_state.personal_data.get('name', 'User')}! You are **NOT Diabetic**\n\n**Confidence Score:** {prediction_proba[0][0]*100:.2f}%")
        is_diabetic = False
    else:
        st.error('⚠️ Important Notice', icon="🔴")
        st.markdown(f"### 💙 Hello {st.session_state.personal_data.get('name', 'User')}! You May Have Diabetes\n\n**Confidence Score:** {prediction_proba[0][1]*100:.2f}%\n\n**Steps:** 1. 🏥 Consult a healthcare professional 2. 📊 Get blood tests")
        is_diabetic = True

    # Lifestyle Impact
    with st.expander("🏃 Lifestyle Impact on Your Score", expanded=True):
        _sign = "+" if _delta >= 0 else ""
        _impact_dir = "increased" if _delta > 0 else ("decreased" if _delta < 0 else "not changed")
        st.caption(f"Your lifestyle **{_impact_dir}** risk by **{_sign}{_delta*100:.1f}%** (from {_raw_diabetic_prob*100:.1f}% → {_adj_diabetic_prob*100:.1f}%).")
        _factor_rows = {
            "🏋️ Exercise": {'Never': +10, '1-2 days': +5, '3-4 days': 0, '5-6 days': -4, 'Daily': -6}.get(_lifestyle.get('exercise_frequency', '3-4 days'), 0),
            "🥗 Diet": {'High Carbs': +8, 'High Fat': +6, 'Low Protein': +3, 'Balanced': 0}.get(_lifestyle.get('diet_type', 'Balanced'), 0),
            "😰 Stress": {'Very High': +8, 'High': +5, 'Moderate': 0, 'Low': -2, 'Very Low': -3}.get(_lifestyle.get('stress_level', 'Moderate'), 0),
            "😴 Sleep": (+7 if _lifestyle.get('sleep_hours', 7) <= 5 else +3 if _lifestyle.get('sleep_hours', 7) == 6 else +2 if _lifestyle.get('sleep_hours', 7) >= 9 else 0),
            "🍷 Alcohol": {'Never': 0, 'Occasionally': +2, 'Regularly': +7}.get(_lifestyle.get('alcohol_consumption', 'Never'), 0),
            "🚭 Smoking": {'Never': 0, 'Former': +2, 'Current': +8}.get(_lifestyle.get('smoking', 'Never'), 0),
        }
        _cols = st.columns(3)
        for idx, (factor, pts) in enumerate(_factor_rows.items()):
            with _cols[idx % 3]:
                _s2 = "+" if pts >= 0 else ""
                st.metric(factor, f"{_s2}{pts}%", delta=pts, delta_color="inverse" if pts <= 0 else "normal")

    st.divider()
    st.subheader("📊 Understanding Your Results - Feature Analysis")
    try:
        feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree', 'Age', 'Diet Type']
        # Fast: use model's built-in feature importances (instant, no shuffling)
        importance_scores = list(model.feature_importances_)
        _is_male = st.session_state.personal_data.get('gender', 'Female') == 'Male'
        diabetes_df = load_diabetes_data()
        X_data = scaler.transform(diabetes_df.drop(columns='Outcome', axis=1))
        if _is_male:
            _preg_idx = feature_names.index('Pregnancies')
            _display_features = [f for f in feature_names if f != 'Pregnancies']
            _display_scores = [importance_scores[i] for i, f in enumerate(feature_names) if f != 'Pregnancies']
            _display_std = np.delete(std_data[0], _preg_idx); _display_X = np.delete(X_data, _preg_idx, axis=1); _display_input = np.delete(input_data[0], _preg_idx)
        else:
            _display_features, _display_scores, _display_std, _display_X, _display_input = feature_names, importance_scores, std_data[0], X_data, input_data[0]
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sorted_idx = np.argsort(_display_scores)
            sorted_names = [_display_features[i] for i in sorted_idx]; sorted_scores = [_display_scores[i] for i in sorted_idx]
            colors = ['#ff6b6b' if _display_std[i] > np.mean(_display_X[:, i]) else '#51cf66' for i in sorted_idx]
            ax.barh(range(len(sorted_idx)), sorted_scores, color=colors)
            ax.set_yticks(range(len(sorted_idx))); ax.set_yticklabels(sorted_names)
            ax.set_xlabel('Importance Score', fontsize=11); ax.set_title('Feature Importance', fontsize=12, fontweight='bold'); ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig); plt.close()
        with col2:
            st.markdown("### Your Personal Risk Assessment:")
            explanation_data = []
            for i, feature in enumerate(_display_features):
                avg_val = np.mean(_display_X[:, i]); is_high = _display_std[i] > avg_val
                if float(_display_scores[i]) > 0.01:
                    explanation_data.append({'factor': feature, 'value': f"{_display_input[i]:.1f}", 'importance': _display_scores[i], 'direction': "⬆️ Increases Risk" if is_high else "⬇️ Reduces Risk"})
            explanation_data.sort(key=lambda x: x['importance'], reverse=True)
            for data in list(itertools.islice(explanation_data, 5)):
                st.markdown(f"**{data['factor']}** ({data['value']})\n{data['direction']} - Score: {data['importance']:.4f}")
    except Exception as e:
        st.warning(f"⚠️ Feature analysis error: {str(e)}")

    st.divider()
    st.subheader("💡 How This Analysis Works")
    st.info("- 🔴 **Red Factors**: Higher than average, increase risk\n- 🟢 **Green Factors**: Lower than average, reduce risk\n\n**Important**: This tool provides insights, not a diagnosis.")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button('← Back to Lifestyle', use_container_width=True, key='page4_back'): st.session_state.page = 3; st.rerun()
    with col3:
        if st.button('Get Personalized Advice →', use_container_width=True, key='page4_next'): st.session_state.page = 'loading_advice'; st.session_state.ai_advice = None; st.rerun()

# Loading Screen — shown while generating AI advice
elif st.session_state.page == 'loading_advice':
    st.markdown("""
    <style>
    /* Hide header, footer, sidebar, and all Streamlit chrome */
    .header-container, header[data-testid="stHeader"], footer,
    [data-testid="stSidebar"], [data-testid="stToolbar"],
    [data-testid="stDecoration"] { display:none !important; }
    .block-container { padding-top:0 !important; max-width:100% !important; }
    @keyframes ldIn{from{opacity:0;transform:scale(.92)}to{opacity:1;transform:scale(1)}}
    @keyframes ldSpin{to{transform:rotate(360deg)}}
    @keyframes ldPulse{0%,100%{opacity:.4;transform:scale(.97)}50%{opacity:1;transform:scale(1.03)}}
    @keyframes ldBar{0%{background-position:-200% center}100%{background-position:200% center}}
    @keyframes ldGlow{0%,100%{box-shadow:0 0 30px rgba(102,126,234,.25)}50%{box-shadow:0 0 60px rgba(118,75,162,.45)}}
    .ld-overlay{position:fixed;inset:0;z-index:999999;background:linear-gradient(135deg,#0a1628 0%,#1a3a52 50%,#0d2340 100%);display:flex;flex-direction:column;align-items:center;justify-content:center;animation:ldIn .6s ease-out}
    .ld-rings{position:relative;width:120px;height:120px;margin-bottom:40px;animation:ldGlow 2.5s ease-in-out infinite}
    .ld-r1{position:absolute;inset:0;border:4px solid rgba(102,126,234,.1);border-top-color:#667eea;border-right-color:#764ba2;border-radius:50%;animation:ldSpin 1.1s cubic-bezier(.68,-.15,.27,1.15) infinite}
    .ld-r2{position:absolute;inset:16px;border:3px solid rgba(118,75,162,.1);border-bottom-color:#764ba2;border-left-color:#ff006e;border-radius:50%;animation:ldSpin .75s linear infinite reverse}
    .ld-dot{position:absolute;inset:36px;background:radial-gradient(circle,#667eea 0%,transparent 70%);border-radius:50%;animation:ldPulse 1.6s ease-in-out infinite}
    .ld-title{color:#fff;font-size:1.6em;font-weight:700;letter-spacing:.5px;margin-bottom:12px;text-align:center;text-shadow:0 2px 12px rgba(102,126,234,.4)}
    .ld-sub{color:#a8b5ff;font-size:1.08em;font-weight:500;animation:ldPulse 2s ease-in-out infinite;text-align:center}
    .ld-track{width:300px;height:4px;background:rgba(102,126,234,.12);border-radius:4px;overflow:hidden;margin-top:30px}
    .ld-bar{width:100%;height:100%;background:linear-gradient(90deg,#667eea,#764ba2,#ff006e,#764ba2,#667eea);background-size:200% 100%;border-radius:4px;animation:ldBar 1.6s linear infinite}
    .ld-note{color:#6c7a96;font-size:.84em;text-align:center;margin-top:20px;line-height:1.7;max-width:380px}
    </style>
    <div class="ld-overlay">
        <div class="ld-rings"><div class="ld-r1"></div><div class="ld-r2"></div><div class="ld-dot"></div></div>
        <div class="ld-title">🤖 Analyzing Your Health Data</div>
        <div class="ld-sub">Generating personalized advice…</div>
        <div class="ld-track"><div class="ld-bar"></div></div>
        <div class="ld-note">Our AI is reviewing your clinical data, lifestyle habits,<br>and nutrition profile to craft a tailored health plan.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── compute prediction & generate advice while spinner is visible ──
    _ld_dl = st.session_state.lifestyle_data.get('diet_type', 'Balanced')
    _ld_de = diet_type_to_model_value(manual_label_to_classifier_label(_ld_dl))
    _ld_inp = np.array([[st.session_state.personal_data.get('pregnancies',0), st.session_state.clinical_data.get('glucose',100), st.session_state.clinical_data.get('blood_pressure',70), st.session_state.clinical_data.get('skin_thickness',20), st.session_state.clinical_data.get('insulin',80), st.session_state.clinical_data.get('bmi',25.0), st.session_state.clinical_data.get('diabetes_pedigree',0.5), st.session_state.personal_data.get('age',30), _ld_de]])
    _ld_std = scaler.transform(_ld_inp)
    _ld_proba = model.predict_proba(_ld_std)[0]
    _ld_delta = lifestyle_risk_delta(st.session_state.lifestyle_data)
    _ld_adj = float(np.clip(float(_ld_proba[1]) + _ld_delta, 0.01, 0.99))
    _ld_pred = 1 if _ld_adj >= 0.5 else 0
    _ld_res = "Diabetic" if _ld_pred == 1 else "Not Diabetic"
    _ld_conf = (_ld_adj if _ld_pred == 1 else 1.0 - _ld_adj) * 100

    _ld_fn = ['Pregnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','Diabetes Pedigree','Age','Diet Type']
    _ld_imp = list(model.feature_importances_)
    _ld_male = st.session_state.personal_data.get('gender','Female') == 'Male'
    _ld_df = load_diabetes_data()
    _ld_X = scaler.transform(_ld_df.drop(columns='Outcome', axis=1))
    if _ld_male:
        _pi = _ld_fn.index('Pregnancies')
        _ld_ft = [f for i,f in enumerate(_ld_fn) if i!=_pi]
        _ld_sc = [s for i,s in enumerate(_ld_imp) if i!=_pi]
        _ld_s = np.delete(_ld_std[0],_pi); _ld_Xf = np.delete(_ld_X,_pi,axis=1); _ld_iv = np.delete(_ld_inp[0],_pi)
    else:
        _ld_ft,_ld_sc,_ld_s,_ld_Xf,_ld_iv = _ld_fn,_ld_imp,_ld_std[0],_ld_X,_ld_inp[0]
    _ld_risks = [{'name':_ld_ft[i],'value':_ld_iv[i],'importance':_ld_sc[i]} for i in range(len(_ld_ft)) if _ld_s[i]>np.mean(_ld_Xf[:,i]) and _ld_sc[i]>0.01]
    _ld_risks.sort(key=lambda x: x['importance'], reverse=True)
    _ld_top = list(itertools.islice(_ld_risks, 4))

    _ld_ps = ", ".join([f"{k}: {v}" for k,v in st.session_state.personal_data.items()])
    _ld_cs = ", ".join([f"{k}: {v}" for k,v in st.session_state.clinical_data.items()])
    _ld_ls = ", ".join([f"{k}: {v}" for k,v in st.session_state.lifestyle_data.items()])
    _ld_rs = ", ".join([f"{r['name']} ({r['value']})" for r in _ld_top]) if _ld_top else "None significant"
    _ld_nutri = st.session_state.get('nutrition_analysis', None)
    _ld_ns = ""
    if _ld_nutri and _ld_nutri.get('calories',0) > 0:
        _ld_ns = f"Daily Intake: {_ld_nutri['calories']:.0f} kcal, Carbs: {_ld_nutri['total_carbs']:.1f}g, Protein: {_ld_nutri['total_protein']:.1f}g, Fat: {_ld_nutri['total_fat']:.1f}g, Diet Type: {_ld_nutri.get('diet_type','Unknown')}"

    _ld_prompt = f"""You are an expert clinical diabetes analyst and lifestyle medicine specialist.
Generate a COMPREHENSIVE, DETAILED health analysis report for this patient. Use specific numbers, targets, and actionable medical-grade recommendations.

PATIENT PROFILE:
- {_ld_ps}

CLINICAL DATA:
- {_ld_cs}

LIFESTYLE:
- {_ld_ls}

NUTRITION ANALYSIS:
- {_ld_ns if _ld_ns else 'Not analyzed'}

ML PREDICTION: {_ld_res} (Confidence: {_ld_conf:.1f}%)
TOP RISK FACTORS: {_ld_rs}

Generate the following sections with rich detail. Use markdown formatting with bold, bullet points, and emojis:

### ⚠️ Clinical Risk Assessment
- Summarize the patient's overall diabetes risk in 3-4 sentences
- Mention specific clinical values that are concerning or healthy
- Reference their BMI and what it means
- Mention family history impact if DPF is elevated

### 🥑 Personalized Nutrition Plan
- Recommend specific daily calorie target based on their BMI and activity level
- List 3-4 specific breakfast, lunch, dinner suggestions (Indian food friendly)
- Mention foods to AVOID and foods to INCREASE
- Target macro split (carbs/protein/fat percentages)

### 🏃 Exercise & Activity Protocol
- Specific exercise recommendations with duration
- Suggest 3 types of exercises suited to their profile
- Weekly schedule suggestion

### 🧘 Stress & Sleep Optimization
- Specific techniques for their stress level
- Sleep hygiene recommendations
- How stress/sleep affects their glucose levels

### 📊 Monitoring Targets
- Recommended glucose monitoring frequency
- Target glucose ranges (fasting and post-meal)
- When to see a doctor / get lab tests
- BMI target if overweight

### 🖼️ Recommended Visuals
List exactly 5 health category names, one per line, from this list ONLY: Healthy Eating Plate, Walking Exercise, Blood Glucose Monitor, Weight Management, Sleep Quality, Yoga and Stretching, Stress Relief Meditation, Cycling Exercise, Blood Pressure Monitor"""

    api_keys = get_gemini_api_keys()
    if api_keys:
        _ld_ok = False
        for _ak in api_keys:
            for _mn in ['gemini-2.0-flash-lite','gemini-2.0-flash','gemini-1.5-flash']:
                try:
                    _cl = genai.Client(api_key=_ak)
                    _rsp = _cl.models.generate_content(model=_mn, contents=_ld_prompt, config=genai.types.GenerateContentConfig(max_output_tokens=1500))
                    st.session_state.ai_advice = _rsp.text; _ld_ok = True; break
                except Exception: continue
            if _ld_ok: break
        if not _ld_ok:
            st.session_state.ai_advice = _generate_local_advice(st.session_state, _ld_res, _ld_conf, _ld_top)
    else:
        st.session_state.ai_advice = _generate_local_advice(st.session_state, _ld_res, _ld_conf, _ld_top)

    st.session_state.page = 5
    st.rerun()

# Page 5: Personalized Advice
elif st.session_state.page == 5:

    _p5_diet_label = st.session_state.lifestyle_data.get('diet_type', 'Balanced')
    _p5_diet_encoded = diet_type_to_model_value(manual_label_to_classifier_label(_p5_diet_label))
    input_data = np.array([[st.session_state.personal_data.get('pregnancies', 0), st.session_state.clinical_data.get('glucose', 100), st.session_state.clinical_data.get('blood_pressure', 70), st.session_state.clinical_data.get('skin_thickness', 20), st.session_state.clinical_data.get('insulin', 80), st.session_state.clinical_data.get('bmi', 25.0), st.session_state.clinical_data.get('diabetes_pedigree', 0.5), st.session_state.personal_data.get('age', 30), _p5_diet_encoded]])
    std_data = scaler.transform(input_data)
    prediction_proba = model.predict_proba(std_data)[0]
    _p5_delta = lifestyle_risk_delta(st.session_state.lifestyle_data)
    _p5_adj_diabetic = float(np.clip(float(prediction_proba[1]) + _p5_delta, 0.01, 0.99))
    prediction = np.array([1 if _p5_adj_diabetic >= 0.5 else 0])
    result_str = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    confidence = (_p5_adj_diabetic if prediction[0] == 1 else 1.0 - _p5_adj_diabetic) * 100

    feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree', 'Age', 'Diet Type']
    # Fast: use model's built-in feature importances (instant, no shuffling)
    importance_scores = list(model.feature_importances_)
    _p5_is_male = st.session_state.personal_data.get('gender', 'Female') == 'Male'
    diabetes_df = load_diabetes_data()
    X_data = scaler.transform(diabetes_df.drop(columns='Outcome', axis=1))
    if _p5_is_male:
        pi = feature_names.index('Pregnancies')
        _p5_features = [f for i, f in enumerate(feature_names) if i != pi]
        _p5_scores = [s for i, s in enumerate(importance_scores) if i != pi]
        _p5_std = np.delete(std_data[0], pi); _p5_X = np.delete(X_data, pi, axis=1); _p5_input = np.delete(input_data[0], pi)
    else:
        _p5_features, _p5_scores, _p5_std, _p5_X, _p5_input = feature_names, importance_scores, std_data[0], X_data, input_data[0]
    risk_factors = [{'name': _p5_features[i], 'value': _p5_input[i], 'importance': _p5_scores[i]} for i in range(len(_p5_features)) if _p5_std[i] > np.mean(_p5_X[:, i]) and _p5_scores[i] > 0.01]
    risk_factors.sort(key=lambda x: x['importance'], reverse=True)
    top_risks = list(itertools.islice(risk_factors, 4))

    personal_str = ", ".join([f"{k}: {v}" for k, v in st.session_state.personal_data.items()])
    clinical_str = ", ".join([f"{k}: {v}" for k, v in st.session_state.clinical_data.items()])
    lifestyle_str = ", ".join([f"{k}: {v}" for k, v in st.session_state.lifestyle_data.items()])
    risk_factors_str = ", ".join([f"{r['name']} ({r['value']})" for r in top_risks]) if top_risks else "None significant"

    # Nutrition analysis context
    _nutri = st.session_state.get('nutrition_analysis', None)
    nutri_str = ""
    if _nutri and _nutri.get('calories', 0) > 0:
        nutri_str = f"Daily Intake: {_nutri['calories']:.0f} kcal, Carbs: {_nutri['total_carbs']:.1f}g, Protein: {_nutri['total_protein']:.1f}g, Fat: {_nutri['total_fat']:.1f}g, Diet Type: {_nutri.get('diet_type', 'Unknown')}"

    prompt = f"""You are an expert clinical diabetes analyst and lifestyle medicine specialist. 
Generate a COMPREHENSIVE, DETAILED health analysis report for this patient. Use specific numbers, targets, and actionable medical-grade recommendations.

PATIENT PROFILE:
- {personal_str}

CLINICAL DATA:
- {clinical_str}

LIFESTYLE:
- {lifestyle_str}

NUTRITION ANALYSIS:
- {nutri_str if nutri_str else 'Not analyzed'}

ML PREDICTION: {result_str} (Confidence: {confidence:.1f}%)
TOP RISK FACTORS: {risk_factors_str}

Generate the following sections with rich detail. Use markdown formatting with bold, bullet points, and emojis:

### ⚠️ Clinical Risk Assessment
- Summarize the patient's overall diabetes risk in 3-4 sentences
- Mention specific clinical values that are concerning or healthy (e.g., "Your glucose at {st.session_state.clinical_data.get('glucose', 100)} mg/dL is...")
- Reference their BMI ({st.session_state.clinical_data.get('bmi', 25)}) and what it means
- Mention family history impact if DPF is elevated

### 🥑 Personalized Nutrition Plan
- Recommend specific daily calorie target based on their BMI and activity level
- List 3-4 specific breakfast, lunch, dinner suggestions (Indian food friendly)
- Mention foods to AVOID and foods to INCREASE
- Target macro split (carbs/protein/fat percentages)

### 🏃 Exercise & Activity Protocol
- Specific exercise recommendations with duration (e.g., "30 min brisk walking, 5 days/week")
- Suggest 3 types of exercises suited to their profile
- Weekly schedule suggestion

### 🧘 Stress & Sleep Optimization
- Specific techniques for their stress level
- Sleep hygiene recommendations
- How stress/sleep affects their glucose levels

### 📊 Monitoring Targets
- Recommended glucose monitoring frequency
- Target glucose ranges (fasting and post-meal)
- When to see a doctor / get lab tests
- BMI target if overweight

### 🖼️ Recommended Visuals
List exactly 5 health category names, one per line, from this list ONLY: Healthy Eating Plate, Walking Exercise, Blood Glucose Monitor, Weight Management, Sleep Quality, Yoga and Stretching, Stress Relief Meditation, Cycling Exercise, Blood Pressure Monitor"""

    with st.spinner("🤖 Getting AI advice..."):
        if st.session_state.ai_advice is None:
            api_keys = get_gemini_api_keys()
            if api_keys:
                success = False; last_error = ""
                _models_to_try = ['gemini-2.0-flash-lite', 'gemini-2.0-flash', 'gemini-1.5-flash']
                for api_key in api_keys:
                    for _model_name in _models_to_try:
                        try:
                            client = genai.Client(api_key=api_key)
                            response = client.models.generate_content(
                                model=_model_name,
                                contents=prompt,
                                config=genai.types.GenerateContentConfig(max_output_tokens=1500)
                            )
                            st.session_state.ai_advice = response.text; success = True; break
                        except Exception as e: last_error = str(e); continue
                    if success: break
                if not success:
                    st.session_state.ai_advice = _generate_local_advice(st.session_state, result_str, confidence, top_risks)
            else:
                st.session_state.ai_advice = _generate_local_advice(st.session_state, result_str, confidence, top_risks)
    ai_advice = st.session_state.ai_advice

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("💡 Your Personalized Health Action Plan")
    st.markdown('</div>', unsafe_allow_html=True)

    import re
    display_text = ai_advice
    image_hints = []
    _vm = re.search(r'###?\s*[^\n]*Recommended Visuals[^\n]*\n+(.*?)(?=\n###|\Z)', ai_advice, flags=re.IGNORECASE | re.DOTALL)
    if _vm:
        display_text = ai_advice[:_vm.start()].strip()
        _raw = re.sub(r'[*_]', '', _vm.group(1)); _raw = re.sub(r'[\n\r]+', ',', _raw); _raw = re.sub(r'-+', ',', _raw)
        for item in _raw.split(','):
            item = item.strip().strip('.')
            if 3 < len(item) <= 60 and 'Score' not in item and ':' not in item: image_hints.append(item)
    st.markdown(display_text)

    if image_hints:
        st.divider(); st.subheader("🖼️ Health & Diabetes Management Guide")
        IMAGE_MAP = {
            # ── Diabetes & clinical monitoring ───────────────────────────────
            'glucose testing':          'images/Glucose_testing.png',
            'glucose monitor':          'images/Glucose_testing.png',
            'blood glucose':            'images/Glucose_testing.png',
            'glucose':                  'images/diabetes_checkup.png',
            'blood sugar':              'images/diabetes_checkup.png',
            'sugar':                    'images/diabetes_checkup.png',
            'diabetes checkup':         'images/diabetes_checkup.png',
            'diabetes':                 'images/diabetes_checkup.png',
            'checkup':                  'images/diabetes_checkup.png',
            'lab test':                 'images/diabetes_checkup.png',
            'hba1c':                    'images/Glucose_testing.png',
            'test':                     'images/Glucose_testing.png',
            'monitor':                  'images/Glucose_testing.png',
            # ── Insulin & medication ──────────────────────────────────────────
            'insulin':                  'images/insulin_injection.png',
            'injection':                'images/insulin_injection.png',
            'medication':               'images/insulin_injection.png',
            'medicine':                 'images/insulin_injection.png',
            'supplement':               'images/insulin_injection.png',
            'pill':                     'images/insulin_injection.png',
            'drug':                     'images/insulin_injection.png',
            # ── Doctor / consultation ─────────────────────────────────────────
            'doctor consultation':      'images/doctor_consultation.png',
            'doctor':                   'images/doctor_consultation.png',
            'physician':                'images/doctor_consultation.png',
            'consultation':             'images/doctor_consultation.png',
            'appointment':              'images/doctor_consultation.png',
            'clinic':                   'images/doctor_consultation.png',
            'healthcare provider':      'images/doctor_consultation.png',
            'medical advice':           'images/doctor_consultation.png',
            # ── Blood pressure & heart ────────────────────────────────────────
            'blood pressure monitor':   'images/bp2.png',
            'blood pressure':           'images/bp1.jpg',
            'pressure':                 'images/bp2.png',
            'bp':                       'images/bp2.png',
            'hypertension':             'images/bp2.png',
            'heart health':             'images/heart_health.png',
            'heart disease':            'images/heart_health.png',
            'cardiovascular':           'images/heart_health.png',
            'heart':                    'images/heart_health.png',
            'cardiac':                  'images/heart_health.png',
            'cholesterol':              'images/heart_health.png',
            # ── Diet & nutrition ──────────────────────────────────────────────
            'healthy eating plate':     'images/healthy_plate.png',
            'eating plate':             'images/healthy_plate.png',
            'plate':                    'images/healthy_plate.png',
            'portion control':          'images/portion_control.png',
            'portion':                  'images/portion_control.png',
            'serving size':             'images/portion_control.png',
            'meal prep':                'images/meal_prep.png' if False else 'images/plate_food.png',
            'eating':                   'images/plate_food.png',
            'meal':                     'images/plate_food.png',
            'nutrition':                'images/healthy_foodplate.png',
            'food':                     'images/plate_food.png',
            'diet':                     'images/healthy_foodplate.png',
            'balanced diet':            'images/healthy_foodplate.png',
            'carb':                     'images/healthy_foodplate.png',
            'calorie':                  'images/healthy_plate.png',
            'protein':                  'images/plate_food.png',
            'fruit':                    'images/fruits_vegetables.png',
            'vegetable':                'images/fruits_vegetables.png',
            'fibre':                    'images/fruits_vegetables.png',
            'fiber':                    'images/fruits_vegetables.png',
            'whole grain':              'images/fruits_vegetables.png',
            'no sugar':                 'images/no_sugar.png',
            'sugar free':               'images/no_sugar.png',
            'avoid sugar':              'images/no_sugar.png',
            'refined':                  'images/no_sugar.png',
            'junk food':                'images/no_sugar.png',
            'hydration':                'images/hydration.png',
            'water':                    'images/hydration.png',
            'fluid':                    'images/hydration.png',
            'drink':                    'images/hydration.png',
            # ── Exercise & activity ───────────────────────────────────────────
            'walking exercise':         'images/walking_exercise.png',
            'brisk walk':               'images/Walking_1.png',
            'morning run':              'images/Walking_1.png',
            'walk':                     'images/walking_exercise.png',
            'jogging':                  'images/Walking_1.png',
            'running':                  'images/Walking_1.png',
            'exercise':                 'images/Walking_1.png',
            'activity':                 'images/walking.png',
            'cardio':                   'images/walking.png',
            'aerobic':                  'images/Walking_1.png',
            'swimming':                 'images/swimming.png',
            'swim':                     'images/swimming.png',
            'pool':                     'images/swimming.png',
            'water exercise':           'images/swimming.png',
            'cycling exercise':         'images/riding_bicycles.png',
            'cycling':                  'images/riding_bicycles.png',
            'bicycle':                  'images/riding_bicycles.png',
            'resistance':               'images/weight_management.png',
            'strength':                 'images/weight_management.png',
            'gym':                      'images/weight_management.png',
            # ── Weight & BMI ──────────────────────────────────────────────────
            'weight management':        'images/weight_management.png',
            'weight':                   'images/weight_management.png',
            'bmi':                      'images/weight_management.png',
            'body':                     'images/weight_management.png',
            'obesity':                  'images/weight_management.png',
            # ── Yoga & stretching ─────────────────────────────────────────────
            'yoga and stretching':      'images/Yoga.png',
            'yoga':                     'images/Yoga.png',
            'stretch':                  'images/health_yoga.png',
            'pilates':                  'images/Yoga.png',
            'flexibility':              'images/health_yoga.png',
            'balance':                  'images/health_yoga.png',
            # ── Stress & meditation ───────────────────────────────────────────
            'stress relief meditation': 'images/stress_relief.png',
            'stress relief':            'images/stress_relief.png',
            'stress':                   'images/stress_relief.png',
            'meditat':                  'images/stress_relief.png',
            'mindful':                  'images/stress_relief.png',
            'relax':                    'images/stress_relief.png',
            'anxiety':                  'images/stress_relief.png',
            'mental':                   'images/stress_relief.png',
            'breath':                   'images/stress_relief.png',
            # ── Sleep ─────────────────────────────────────────────────────────
            'sleep quality':            'images/Sleep.png',
            'sleep':                    'images/Sleep.png',
            'sound sleep':              'images/sound_sleep.png',
            'rest':                     'images/sleeping.png',
            'insomnia':                 'images/sound_sleep.png',
            'recovery':                 'images/sleeping.png',
            # ── Lifestyle risks ───────────────────────────────────────────────
            'smoking':                  'images/smoking_alcohol.png',
            'alcohol':                  'images/smoking_alcohol.png',
            'tobacco':                  'images/smoking_alcohol.png',
            'quit smoking':             'images/smoking_alcohol.png',
            'avoid alcohol':            'images/smoking_alcohol.png',
            'habit':                    'images/smoking_alcohol.png',
        }
        ALL_IMAGES = list(dict.fromkeys(IMAGE_MAP.values()))
        def get_local_image(hint, used, fb=0):
            hl = hint.lower()
            # Try longest keyword match first for precision
            for kw in sorted(IMAGE_MAP, key=len, reverse=True):
                if kw in hl and IMAGE_MAP[kw] not in used: return IMAGE_MAP[kw]
            for img in ALL_IMAGES:
                if img not in used: return img
            return ALL_IMAGES[fb % len(ALL_IMAGES)]
        used_images = set()
        # Show up to 6 images: row 1 = first 3, row 2 = next 3
        for row_hints in [image_hints[:3], image_hints[3:6]]:
            if not row_hints: continue
            cols = st.columns(len(row_hints))
            for idx, (col, hint) in enumerate(zip(cols, row_hints)):
                with col:
                    local_img = get_local_image(hint, used_images, idx); used_images.add(local_img)
                    try:
                        st.image(local_img, use_container_width=True)
                        st.markdown(f'<div style="text-align:center;font-size:0.85em;font-weight:600;color:#a8b5ff;margin-top:-6px;">💡 {hint}</div>', unsafe_allow_html=True)
                    except Exception: st.caption(f"📷 {hint}")

    st.divider()
    # PDF Report
    import re as _re
    def _ascii(s): return _re.sub(r'[^\x00-\x7F]+', '', str(s))
    def _status(v, lo, hi): return "LOW" if v < lo else ("HIGH" if v > hi else "NORMAL")
    clinical = st.session_state.clinical_data; personal = st.session_state.personal_data
    param_table = [("Glucose", clinical.get('glucose',0), "mg/dL", 70, 100), ("Blood Pressure", clinical.get('blood_pressure',0), "mm Hg", 60, 80), ("BMI", clinical.get('bmi',0), "", 18.5, 24.9), ("Insulin", clinical.get('insulin',0), "mu U/ml", 16, 166), ("Skin Thickness", clinical.get('skin_thickness',0), "mm", 10, 50), ("Diabetes Pedigree", clinical.get('diabetes_pedigree',0), "", 0, 0.5)]
    clean_advice = ""
    if ai_advice:
        clean_advice = _ascii(ai_advice)
        clean_advice = _re.sub(r'[*_`#]', '', clean_advice)
        if "Recommended Visuals" in clean_advice: clean_advice = clean_advice.split("Recommended Visuals")[0]
        clean_advice = clean_advice.strip()
        if len(clean_advice) > 380:
            cut = clean_advice.rfind('.', 0, 380)
            clean_advice = clean_advice[:cut+1 if cut != -1 else 380].strip() + "..."

    # fpdf2 v2 compatible — new_x/new_y replace ln=True; output() returns bytes directly
    _NX, _NY = "LMARGIN", "NEXT"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_margins(10, 10, 10)

    # Title
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 6, "GlucoFit AI - Diabetes Risk Report", align="C", new_x=_NX, new_y=_NY)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.cell(0, 4, f"Generated: {pd.Timestamp.now().strftime('%d %b %Y %H:%M')}", align="C", new_x=_NX, new_y=_NY)
    pdf.ln(1); pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(2)

    # Patient info + risk side by side
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(95, 4, "PATIENT INFORMATION")
    pdf.cell(0, 4, "RISK ASSESSMENT", new_x=_NX, new_y=_NY)
    _left  = [("Name", _ascii(personal.get('name','N/A'))), ("Age", f"{personal.get('age','N/A')} yrs"), ("Gender", personal.get('gender','N/A'))]
    _right = [("Prediction", result_str), ("Confidence", f"{confidence:.1f}%"), ("Risk Level", "HIGH" if prediction[0]==1 else "LOW")]
    for (lk, lv), (rk, rv) in zip(_left, _right):
        pdf.set_font("Helvetica", "B", 7.5); pdf.cell(20, 3.5, f"{lk}:")
        pdf.set_font("Helvetica", "", 7.5);  pdf.cell(75, 3.5, lv)
        pdf.set_font("Helvetica", "B", 7.5); pdf.cell(20, 3.5, f"{rk}:")
        pdf.set_font("Helvetica", "", 7.5);  pdf.cell(0, 3.5, rv, new_x=_NX, new_y=_NY)
    pdf.ln(2)

    # Clinical parameters table
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(0, 4, "CLINICAL PARAMETERS", new_x=_NX, new_y=_NY)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(1)
    pdf.set_fill_color(220, 220, 220); pdf.set_font("Helvetica", "B", 7.5)
    pdf.cell(56, 4.5, "Parameter",    border=1, fill=True)
    pdf.cell(36, 4.5, "Value",        border=1, fill=True, align="C")
    pdf.cell(52, 4.5, "Normal Range", border=1, fill=True, align="C")
    pdf.cell(0,  4.5, "Status",       border=1, fill=True, align="C", new_x=_NX, new_y=_NY)
    _sc = {"NORMAL": (180,230,180), "HIGH": (255,200,180), "LOW": (200,210,255)}
    pdf.set_font("Helvetica", "", 7.5)
    for nm, val, unit, lo, hi in param_table:
        s = _status(float(val), lo, hi)
        pdf.set_fill_color(255,255,255)
        pdf.cell(56, 4.5, nm, border=1)
        pdf.cell(36, 4.5, f"{val:.1f} {unit}".strip(), border=1, align="C")
        pdf.cell(52, 4.5, f"{lo}-{hi} {unit}".strip(), border=1, align="C")
        pdf.set_fill_color(*_sc.get(s,(255,255,255)))
        pdf.cell(0, 4.5, s, border=1, align="C", fill=True, new_x=_NX, new_y=_NY)
    pdf.ln(2)

    # AI Recommendations
    if clean_advice.strip():
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(0, 4, "AI RECOMMENDATIONS", new_x=_NX, new_y=_NY)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(1)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.multi_cell(0, 3.8, clean_advice.strip()); pdf.ln(2)

    # Recommended Next Steps
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(0, 4, "RECOMMENDED NEXT STEPS", new_x=_NX, new_y=_NY)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(1.5)
    _is_high_risk = prediction[0] == 1
    _bmi_val      = clinical.get('bmi', 25)
    _glucose_val  = clinical.get('glucose', 100)
    _steps = []
    if _is_high_risk:
        _steps.append("[ ] Urgent: See a doctor within 2 weeks for a comprehensive diabetes screening.")
        _steps.append("[ ] Request HbA1c and fasting glucose tests for confirmation.")
    else:
        _steps.append("[ ] Schedule a routine health check-up within the next 3-6 months.")
    if _glucose_val > 100:
        _steps.append(f"[ ] Monitor glucose - level ({_glucose_val} mg/dL) exceeds ideal fasting range (70-100 mg/dL).")
    if _bmi_val >= 25:
        _steps.append(f"[ ] Work toward healthy BMI (18.5-24.9) - current BMI {_bmi_val:.1f}.")
    _steps.append("[ ] Eat a balanced diet and exercise 30+ minutes daily.")
    _steps = _steps[:4]  # cap at 4 to save space
    _steps.append("[ ] Share this report with your healthcare provider for guidance.")
    pdf.set_font("Helvetica", "", 7.5)
    for _step in _steps:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(180, 4, f"  {_step}")
    pdf.ln(2)

    # Doctor's Recommendations
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(0, 4, "DOCTOR'S RECOMMENDATIONS", new_x=_NX, new_y=_NY)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(1.5)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.cell(0, 4, "(To be completed by the attending physician)", new_x=_NX, new_y=_NY)
    pdf.ln(2)
    for _ in range(4):                        # 4 blank ruled lines
        pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(6)
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.cell(95, 4, "Doctor's Name: _____________________________")
    pdf.cell(0,  4, "Date: _____________________", new_x=_NX, new_y=_NY)
    pdf.ln(3)
    pdf.cell(95, 4, "Signature: _________________________________")
    pdf.cell(0,  4, "Registration No: _______________", new_x=_NX, new_y=_NY)
    pdf.ln(4)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 6)
    pdf.multi_cell(0, 3, ("DISCLAIMER: This report is AI-generated for informational purposes only. "
        "It does not constitute a medical diagnosis or professional medical advice. "
        "Please consult a qualified healthcare professional for clinical evaluation and treatment decisions."))

    # Output — fpdf2 v2: pdf.output() returns bytes directly
    pdf_buf = io.BytesIO(pdf.output())
    st.download_button(label="📥 Download PDF Report", data=pdf_buf, file_name="glucofit_report.pdf", mime="application/pdf", use_container_width=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button('← Back to Results', use_container_width=True, key='page5_back'): st.session_state.page = 4; st.rerun()
    with col2:
        if st.button('🔄 Start Over', use_container_width=True, key='page5_restart'):
            st.session_state.page = 1; st.session_state.personal_data = {}; st.session_state.clinical_data = {}; st.session_state.lifestyle_data = {}; st.session_state.ai_advice = None; st.session_state.insulin_estimated = False; st.rerun()
    with col3:
        if st.button('🏠 Home', use_container_width=True, key='page5_home'): st.switch_page("app.py")
