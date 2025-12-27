import streamlit as st
import requests
import json
import time

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
api_url = "http://127.0.0.1:8000"     

st.set_page_config(
    page_title="Smart Agro AI",
    layout="wide",
    initial_sidebar_state="auto"
)


# -------------------------------------------------
# DARK THEME + MODERN UI CSS
# -------------------------------------------------
st.markdown("""
<style>
/* Background and layout */
.stApp {
    background: radial-gradient(circle at top right, #141e30 0%, #0a0f16 45%, #000 100%);
    color: #E0E6ED;
    font-family: 'Segoe UI', sans-serif;
}
.block-container {
    padding-top: 2.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Header */
.header-title{
    padding-top: 2.5rem;
    font-size:2.8rem;
    font-weight:850;
    color:#00e676;
    text-align:center;
    letter-spacing: 1px;
    margin-top: -10px;
    text-shadow: 0 0 18px rgba(0,255,136,0.4);
}
.subtext {
    font-size:1.2rem;
    color:#9fb4c7;
    text-align:center;
    margin-bottom:2.2rem;
}

/* Cards */
.result-box {
    background:#0F161E;
    padding:2rem;
    border-radius:18px;
    border:1px solid rgba(0,255,136,0.18);
    box-shadow:0 0 22px rgba(0,255,136,0.12);
    margin-top:1.3rem;
    transition: all .3s ease;
}
.result-box:hover {
    box-shadow:0 0 32px rgba(0,255,136,0.35);
}

/* Tabs spacing */
.stTabs [role="tablist"] {
    margin-bottom: 2rem;
}

/* Buttons */
.stButton > button {
    background:#00e676;
    color:#000;
    border-radius:10px;
    padding:0.8rem 1.2rem;
    font-size:1.05rem;
    border:none;
    font-weight:600;
    width:100%;
    cursor:pointer;
    transition:all .25s ease;
}
.stButton > button:hover {
    background:#00c853;
    transform: scale(1.01);
    box-shadow:0 0 18px rgba(0,255,136,0.30);
}

/* Badges */
.info-badge {
    display:inline-block;
    background:rgba(0,255,136,0.14);
    color:#00e676;
    padding:0.45em 0.85em;
    border-radius:8px;
    margin:0.2em;
    font-size:1.05em;
    border:1px solid rgba(0,255,136,0.25);
}

/* Image preview styling */
img {
    border-radius:12px;
    border:1px solid rgba(255,255,255,0.1);
    box-shadow:0 0 15px rgba(0, 0, 0, .45);
    margin-top:1rem;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("<div class='header-title'>🌱 Smart Agro AI</div>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Wind speed prediction & sugarcane leaf disease detection</p>", unsafe_allow_html=True)



# =================================================
#                   TABS
# =================================================
tab1, tab2 = st.tabs(["💨 Wind Speed Prediction", "🍃 Sugarcane Disease Detection"])



# =================================================
# TAB 1 — WIND SPEED PREDICTION
# =================================================
with tab1:
    st.markdown("## 💨 Predict Wind Speed")

    col_left, col_right = st.columns([1,1])
    with col_left:
        day = st.number_input("Day", min_value=1, max_value=31, value=30)
        minute = st.number_input("Minute", value=0)
        glo_avg = st.number_input("Global Avg", value=-1.818)
        diff_avg = st.number_input("Diffuse Avg", value=-1.805)
        lw_avg = st.number_input("LW Avg", value=357.3)

    with col_right:
        tp_sfc = st.number_input("Surface Temp", value=19.21)
        humid = st.number_input("Humidity", value=88.9)
        press = st.number_input("Pressure", value=905.42)
        wd_10m = st.number_input("Wind Direction", value=50.6)

    wind_features = [day, minute, glo_avg, diff_avg, lw_avg, tp_sfc, humid, press, wd_10m]

    if st.button("🔎 Predict Wind Speed"):
        try:
            with st.spinner("Connecting to AI model..."):
                payload = {"features": wind_features}
                response = requests.post(f"{api_url}/predict/wind", json=payload)
                
                if response.status_code == 200:
                    pred = response.json()["predicted_wind_speed"]

                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.markdown(f"### 🌬 Predicted Wind Speed: **{pred:.3f} m/s**")
                    st.markdown(f"<span class='info-badge'>Model: temp2015_2</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"API error: {response.status_code}")
        except Exception as e:
            st.error(str(e))



# =================================================
# TAB 2 — LEAF IMAGE DISEASE DETECTION
# =================================================
with tab2:
    st.markdown("## 🍃 Sugarcane Leaf Disease Detection")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        col_p, col_blank = st.columns([1,1])
        with col_p:
            st.image(uploaded_file, caption="Uploaded Leaf", width=350)

    if uploaded_file and st.button("🧪 Detect Disease"):
        try:
            with st.spinner("Analyzing image using deep learning model..."):
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                response = requests.post(f"{api_url}/predict/image", files=files)

                if response.status_code == 200:
                    result = response.json()
                    label = result["predicted_label"]
                    conf = result["confidence"] * 100

                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.markdown(f"### 🌿 Disease: **{label}**")
                    st.markdown(f"<span class='info-badge'>Confidence: {conf:.2f}%</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.error(f"API error: {response.status_code}")
        except Exception as e:
            st.error(str(e))
